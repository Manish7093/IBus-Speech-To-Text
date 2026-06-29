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

try:
    from moonshine_voice import (
        Transcriber,
        TranscriptEventListener,
        ModelArch,
    )
    MOONSHINE_AVAILABLE = True
except ImportError:
    LOG_MSG.warning("moonshine_voice not available. Install with: pip install moonshine-voice")
    MOONSHINE_AVAILABLE = False
    TranscriptEventListener = object

class _LineListener(TranscriptEventListener):

    def __init__(self, engine):
        if MOONSHINE_AVAILABLE:
            super().__init__()
        self._engine = engine
        self._emitted_lines = []

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
            GLib.idle_add(self._engine._emit_text, text)

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
        self._tx_lock = threading.Lock()
        self._session_active = False
        self._stopping = False

        self._process_queue = queue.Queue()
        self._stop_processing = False
        self._use_partial_results = False
        self._process_thread = threading.Thread(target=self._process_worker, daemon=True)
        self._process_thread.start()
        self._set_model()

    def __del__(self):
        try:
            if LOG_MSG is not None:
                LOG_MSG.info("Moonshine __del__")
            self._stop_processing = True
            if self._process_thread is not None:
                self._process_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            super().__del__()
        except (AttributeError, TypeError):
            pass

    def destroy(self):
        self._stop_processing = True
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)

        self._current_locale.disconnect(self._locale_id)
        self._locale_id = 0

        if self._model_id != 0:
            self._model.disconnect(self._model_id)
            self._model_id = 0

        with self._tx_lock:
            self._teardown_transcriber()

        if self._appsink is not None and getattr(self, "_new_sample_id", 0) !=0:
            self._appsink.disconnect(self._new_sample_id)
            self._new_sample_id = 0
        self._appsink = None
        LOG_MSG.info("Moonshine.destroy() called")
        super().destroy()

    def _teardown_transcriber(self):
        if self._transcriber is None:
            return
        try:
            if self._session_active:
                self._transcriber.stop()
        except Exception as e:
            LOG_MSG.debug("error stopping transcriber on teardown: %s", e)
        self._session_active = False
        self._transcriber = None
        self._listener = None

    def _load_moonshine_model(self, model_path, model_arch):
        if not MOONSHINE_AVAILABLE:
            LOG_MSG.error("moonshine_voice not available")
            return False
        try:
            arch = model_arch if model_arch is not None else ModelArch.BASE
            LOG_MSG.info("Loading Moonshine model: %s (arch=%s)", model_path, arch)
            transcriber = Transcriber(model_path=model_path, model_arch=arch)
            listener = _LineListener(self)
            transcriber.add_listener(listener)
            with self._tx_lock:
                self._teardown_transcriber()
                self._transcriber = transcriber
                self._listener = listener
            return True
        except Exception as e:
            LOG_MSG.error("Failed to load Moonshine model: %s", e)
            with self._tx_lock:
                self._teardown_transcriber()
            return False

    def _set_model_path(self):
        if self._model is None or self._model.available() is False:
            LOG_MSG.info("model path does not exist (%s - %s)",
                         self._model.get_name() if self._model else "None",
                         self._model.get_path() if self._model else "None")
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

        success = self._load_moonshine_model(new_model_path, new_model_arch)

        if state >= Gst.State.READY:
            self.pipeline.set_state(state)

        if success:
            self.emit("model-changed")

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

        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        audio_data = np.frombuffer(map_info.data, dtype=np.int16)
        buf.unmap(map_info)

        audio_float = audio_data.astype(np.float32) / 32768.0
        self._process_queue.put(("audio", audio_float))

        return Gst.FlowReturn.OK

    def _process_worker(self):
        # Feed queued audio into the Moonshine transcriber.
        while not self._stop_processing:
            try:
                cmd, payload = self._process_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            pending_done = 1
            audio_batch = None
            stop_requested = False

            if cmd == "audio":
                audio_batch = [payload]
                while True:
                    try:
                        ncmd, npayload = self._process_queue.get_nowait()
                    except queue.Empty:
                        break
                    pending_done += 1
                    if ncmd == "audio":
                        audio_batch.append(npayload)
                    else:
                        stop_requested = True
                        break
            elif cmd == "stop":
                stop_requested = True

            try:
                with self._tx_lock:
                    transcriber = self._transcriber
                    if transcriber is not None:
                        if audio_batch:
                            if not self._session_active:
                                transcriber.start()
                                if self._listener is not None:
                                    self._listener.reset()
                                self._session_active = True
                            chunk = (audio_batch[0] if len(audio_batch) == 1
                                     else np.concatenate(audio_batch))
                            transcriber.add_audio(chunk, SAMPLE_RATE)
                        if stop_requested and self._session_active:
                            transcriber.stop()
                            self._session_active = False
            except Exception as e:
                LOG_MSG.error("Moonshine processing error: %s", e, exc_info=True)
            finally:
                for _ in range(pending_done):
                    self._process_queue.task_done()


    def _emit_text(self, text):
        self.emit("text", text)
        return False

    def get_final_results(self):
        if self._stop_processing or self._process_thread is None or not self._process_thread.is_alive():
            return
        self._process_queue.put(("stop", None))
        self._process_queue.join()

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
