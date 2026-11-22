import logging
import threading
import queue
import numpy as np

from gi.repository import Gst, GLib
from sttutils import *
from sttgstbase import STTGstBase
from sttcurrentlocale import stt_current_locale
from sttmoonshinemodel import STTMoonshineModel

LOG_MSG = logging.getLogger()

SAMPLE_RATE = 16000

_LISTENER_CLASS = None

def _line_listener_class():
    #Build the listener subclass on first use
    global _LISTENER_CLASS
    if _LISTENER_CLASS is not None:
        return _LISTENER_CLASS

    from moonshine_voice import TranscriptEventListener

    class _LineListener(TranscriptEventListener):

        def __init__(self, engine):
            super().__init__()
            self._engine = engine
            self._emitted_lines = []
            self.session = 0

        def reset(self):
            self._emitted_lines.clear()

        def on_line_completed(self, event):
            line = event.line
            if any(line is seen for seen in self._emitted_lines):
                return
            self._emitted_lines.append(line)
            text = (line.text or "").strip()
            if text:
                LOG_MSG.info("Moonshine transcription result: '%s'", text)
                GLib.idle_add(self._engine._emit_text, text, self.session)

    _LISTENER_CLASS = _LineListener
    return _LISTENER_CLASS


class STTGstMoonshine(STTGstBase):
    __gtype_name__ = 'STTGstMoonshine'
    _pipeline_def = "pulsesrc blocksize=3200 buffer-time=9223372036854775807 ! " \
                    "audio/x-raw,format=S16LE,rate=16000,channels=1 ! " \
                    "webrtcdsp noise-suppression-level=3 echo-cancel=false ! " \
                    "queue ! " \
                    "appsink name=MoonshineSink emit-signals=true sync=false"

    _pipeline_def_alt = "pulsesrc blocksize=3200 buffer-time=9223372036854775807 ! " \
                        "audio/x-raw,format=S16LE,rate=16000,channels=1 ! " \
                        "queue ! " \
                        "appsink name=MoonshineSink emit-signals=true sync=false"

    def __init__(self, current_locale=None):
        plugin = Gst.Registry.get().find_plugin("webrtcdsp")
        if plugin is not None:
            super().__init__(pipeline_definition=STTGstMoonshine._pipeline_def)
            LOG_MSG.debug("using Webrtcdsp plugin")
        else:
            super().__init__(pipeline_definition=STTGstMoonshine._pipeline_def_alt)
            LOG_MSG.debug("not using Webrtcdsp plugin")

        if self.pipeline is None:
            LOG_MSG.error("pipeline was not created")
            return

        self._appsink = self.pipeline.get_by_name("MoonshineSink")
        if self._appsink is None:
            LOG_MSG.error("no appsink element!")
            return

        self._on_new_sample_id = self._appsink.connect("new-sample", self._on_new_sample)

        if current_locale is None:
            self._current_locale = stt_current_locale()
        else:
            self._current_locale = current_locale

        self._locale_id = self._current_locale.connect("changed", self._locale_changed)

        self._model_id = 0
        self._model = None
        self._transcriber = None
        self._listener = None
        self._load_generation = 0
        self._tx_lock = threading.Lock()
        self._session_active = False
        self._stream_session = 0
        self._stopping = False
        self._destroyed = False

        self._process_queue = queue.Queue()
        self._stop_processing = False
        self._use_partial_results = False
        self._process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self._process_thread.start()
        self._set_model()

    def __del__(self):
        try:
            self._stop_processing = True
            thread = self._process_thread
            self._process_thread = None
            if thread is not None:
                thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            super().__del__()
        except (AttributeError, TypeError):
            pass

    def destroy(self):
        if self._destroyed:
            LOG_MSG.debug("Moonshine engine already destroyed")
            return
        self._destroyed = True

        self._stop_processing = True
        self._load_generation += 1

        thread = self._process_thread
        self._process_thread = None
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                LOG_MSG.warning("Moonshine worker thread did not stop in time")

        if self._locale_id != 0:
            self._current_locale.disconnect(self._locale_id)
            self._locale_id = 0

        if self._model_id != 0 and self._model is not None:
            self._model.disconnect(self._model_id)
            self._model_id = 0

        with self._tx_lock:
            self._teardown_transcriber()

        if self._appsink is not None and self._on_new_sample_id != 0:
            self._appsink.disconnect(self._on_new_sample_id)
            self._on_new_sample_id = 0
        self._appsink = None

        LOG_MSG.info("Moonshine.destroy() called")
        super().destroy()

    def _teardown_transcriber(self):
        if self._transcriber is None:
            return
        try:
            if self._session_active:
                self._set_listener_session(self._stream_session)
                self._transcriber.stop()
        except Exception as e:
            LOG_MSG.debug("error stopping transcriber on teardown: %s", e)
        self._session_active = False
        self._transcriber = None
        self._listener = None

    def _load_moonshine_model(self, model_path, model_arch):
        # Returns a (transcriber, listener) pair on success
        try:
            from moonshine_voice import ModelArch, Transcriber
        except Exception as e:
            LOG_MSG.error("moonshine_voice not available (%s). "
                          "Install with: pip install moonshine-voice", e)
            return (None, None)

        try:
            arch = (ModelArch(model_arch) if model_arch is not None else ModelArch.BASE)
            LOG_MSG.info("Loading Moonshine model: %s (arch=%s)", model_path, arch)
            transcriber = Transcriber(model_path=model_path, model_arch=arch)
            listener = _line_listener_class()(self)
            transcriber.add_listener(listener)
            return (transcriber, listener)
        except Exception as e:
            LOG_MSG.error("Failed to load Moonshine model: %s", e)
            return (None, None)

    def _load_model_thread(self, model_path, model_arch, restore_state, generation):
    # Load the model in the worker and schedule completion on the main loop
    # @model_path: Path to the selected model
    # @model_arch: Selected Moonshine arch
    # @restore_state: Pipeline state before loading
    # @generation: ID of the model load request
        transcriber, listener = self._load_moonshine_model(model_path, model_arch)
        GLib.idle_add(self._model_load_done_cb,
                      transcriber, listener, restore_state, generation)

    def _model_load_done_cb(self, transcriber, listener, restore_state, generation):
    # Apply the model load result if the request is still current
    # @transcriber: Success -> Moonshine transcriber, failure -> None
    # @listener: Success -> Listener paired with the transcriber, failure -> None
    # @restore_state: Pipeline state before loading
    # @generation: ID of the model load request
        if self.pipeline is None or generation != self._load_generation:
            LOG_MSG.debug("discarding stale model load")
            return False

        with self._tx_lock:
            self._teardown_transcriber()
            self._transcriber = transcriber
            self._listener = listener

        if transcriber is None:
            return False

        self.restore_state_after_load(restore_state)

        self.emit("model-changed")
        return False

    def _set_model_path(self):
        if self._model is None or self._model.available() is False:
            LOG_MSG.info("model path does not exist (%s - %s)",
                         self._model.get_name() if self._model else "None",
                         self._model.get_path() if self._model else "None")
            self._load_generation += 1
            with self._tx_lock:
                self._teardown_transcriber()
            self.emit("model-changed")
            return

        new_model_path = self._model.get_path()
        new_model_arch = self._model.get_arch()
        LOG_MSG.debug("model ready %s", new_model_path)

        ret, state, pending = self.pipeline.get_state(0)
        if state >= Gst.State.READY:
            self.pipeline.set_state(Gst.State.READY)

        # Load off the main loop to keep IBus responsive
        self._load_generation += 1
        with self._tx_lock:
            self._teardown_transcriber()

        threading.Thread(target=self._load_model_thread,
                         args=(new_model_path, new_model_arch, state,
                               self._load_generation),
                         daemon=True).start()

    def _model_changed(self, model):
        self._set_model_path()

    def _set_model(self):
        if (self._model is not None and
                self._model.get_locale() == self._current_locale.locale):
            return

        if self._model_id != 0:
            self._model.disconnect(self._model_id)
            self._model_id = 0

        self._model = STTMoonshineModel(locale_str=self._current_locale.locale)
        self._model_id = self._model.connect("changed", self._model_changed)
        self._set_model_path()

    def _locale_changed(self, locale):
        self._set_model()

    def _on_new_sample(self, appsink):
        if getattr(self, "_stopping", False):
            return Gst.FlowReturn.OK

        session = self._session_id

        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        if self.audio_is_stale(sample):
            LOG_MSG.debug("dropping audio recorded before the focus changed")
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        audio_data = np.frombuffer(map_info.data, dtype=np.int16)
        buf.unmap(map_info)

        audio_float = audio_data.astype(np.float32) / 32768.0

        if session != self._session_id:
            LOG_MSG.debug("focus changed while this audio was read, "
                          "dropping it")
            return Gst.FlowReturn.OK

        self._process_queue.put(("audio", audio_float, session))

        return Gst.FlowReturn.OK

    def _process_worker(self):
        # Feed queued audio into the Moonshine transcriber.
        carried = None
        while not self._stop_processing:
            if carried is not None:
                cmd, payload, session = carried
                carried = None
            else:
                try:
                    cmd, payload, session = self._process_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                self._process_queue.task_done()

            audio_batch = None
            stop_requested = False

            if cmd == "audio":
                audio_batch = [payload]
                while True:
                    try:
                        item = self._process_queue.get_nowait()
                    except queue.Empty:
                        break
                    self._process_queue.task_done()
                    if item[2] != session:
                        # Keep audio from different sessions in separate chunks
                        carried = item
                        break
                    if item[0] == "audio":
                        audio_batch.append(item[1])
                    else:
                        stop_requested = True
                        break
            elif cmd == "stop":
                stop_requested = True

            stale = bool(session != self._session_id)
            if stale and stop_requested is False:
                LOG_MSG.debug("dropping audio captured for session %i", session)
                continue

            if stale:
                audio_batch = None

            try:
                with self._tx_lock:
                    transcriber = self._transcriber
                    if transcriber is not None:
                        if audio_batch:
                            # Flush the previous stream under its original session
                            if self._session_active and \
                               self._stream_session != session:
                                self._set_listener_session(self._stream_session)
                                transcriber.stop()
                                self._session_active = False

                            if not self._session_active:
                                transcriber.start()
                                if self._listener is not None:
                                    self._listener.reset()
                                self._session_active = True
                                self._stream_session = session
                            self._set_listener_session(self._stream_session)
                            chunk = (audio_batch[0] if len(audio_batch) == 1
                                     else np.concatenate(audio_batch))
                            transcriber.add_audio(chunk, SAMPLE_RATE)

                        if stop_requested and self._session_active and \
                           self._stream_session == session:
                            self._set_listener_session(self._stream_session)
                            transcriber.stop()
                            self._session_active = False
            except Exception as e:
                LOG_MSG.error("Moonshine processing error: %s", e, exc_info=True)


    def _set_listener_session(self, session):
    # Tag synchronous listener callbacks while holding _tx_lock
        if self._listener is not None:
            self._listener.session = session

    def _emit_text(self, text, session):
        if session != self._session_id:
            LOG_MSG.debug("dropping result of session %i (current is %i): '%s'",
                          session, self._session_id, text)
            return False

        self.emit("text", text)
        return False

    def _worker_is_alive(self):
        return bool(self._stop_processing is False and
                    self._process_thread is not None and
                    self._process_thread.is_alive())

    def discard_pending(self):
    # Invalidate the session and close its Moonshine stream.
        stale = self._session_id
        super().discard_pending()
        # Let the worker reject stale jobs so newly queued audio survives
        if self._worker_is_alive():
            self._process_queue.put(("stop", None, stale))

    def get_final_results(self):
        # Let the worker finish asynchronously so IBus can handle key events
        if self._worker_is_alive() is False:
            return
        self._process_queue.put(("stop", None, self._session_id))

    def get_results(self):
        pass

    def set_use_partial_results(self, active):
        self._use_partial_results = active

    def set_alternatives_num(self, num):
        pass

    def has_model(self):
        if self._model is None or self._model.available() is False:
            return False
        return super().has_model()

    def _stop_real(self):
        self._stopping = True
        try:
            self.get_final_results()
            return super()._stop_real()
        finally:
            self._stopping = False
