import logging
import threading
import numpy as np
import re
import os
import time

from collections import deque
from pathlib import Path
from gi.repository import Gst, GLib
from sttutils import *
from sttgstbase import STTGstBase
from sttcurrentlocale import stt_current_locale
from sttwhispermodel import STTWhisperModel

LOG_MSG = logging.getLogger()
SPECIAL_PATTERN = re.compile(r'^(?:\[[^\]]+\]|\([^)]+\))$',re.IGNORECASE)

try:
    from pywhispercpp.model import Model
    WHISPER_AVAILABLE = True
except ImportError:
    LOG_MSG.warning("pywhispercpp not available. Install with: pip install pywhispercpp")
    WHISPER_AVAILABLE = False
try:
    from sttvad import STTVad
    VAD_MODULE_OK = True
except ImportError:
    LOG_MSG.warning("sttvad module not found - VAD disabled")
    VAD_MODULE_OK = False

WINDOW_SECONDS      = 5.0
MIN_SAMPLES         = 1024
SAMPLE_RATE         = 16000
AUDIO_CTX           = 768
TEMPERATURE_INC     = -1.0
MAX_TOKENS          = 0
_cpu_count = os.cpu_count() or 1
N_THREADS           = max(4, _cpu_count // 2 )
MAX_RING_QUEUE_DEPTH = 2

class STTGstWhisper(STTGstBase):
    __gtype_name__ = 'STTGstWhisper'
    _pipeline_def = "pulsesrc blocksize=3200 buffer-time=9223372036854775807 ! " \
                    "audio/x-raw,format=S16LE,rate=16000,channels=1 ! " \
                    "webrtcdsp noise-suppression-level=3 echo-cancel=false ! " \
                    "queue ! " \
                    "appsink name=WhisperSink emit-signals=true sync=false"

    _pipeline_def_alt = "pulsesrc blocksize=3200 buffer-time=9223372036854775807 ! " \
                        "audio/x-raw,format=S16LE,rate=16000,channels=1 ! " \
                        "queue ! " \
                        "appsink name=WhisperSink emit-signals=true sync=false"

    def __init__(self, current_locale=None):
        plugin = Gst.Registry.get().find_plugin("webrtcdsp")
        if plugin is not None:
            super().__init__(pipeline_definition=STTGstWhisper._pipeline_def)
            LOG_MSG.debug("using Webrtcdsp plugin")
        else:
            super().__init__(pipeline_definition=STTGstWhisper._pipeline_def_alt)
            LOG_MSG.debug("not using Webrtcdsp plugin")

        if self.pipeline is None:
            LOG_MSG.error("pipeline was not created")
            return

        self._appsink = self.pipeline.get_by_name("WhisperSink")
        if self._appsink is None:
            LOG_MSG.error("no appsink element!")
            return

        self._appsink.connect("new-sample", self._on_new_sample)

        if current_locale is None:
            self._current_locale = stt_current_locale()
        else:
            self._current_locale = current_locale

        self._locale_id = self._current_locale.connect("changed", self._locale_changed)

        self._model_id = 0
        self._model    = None
        self._whisper  = None
        self._vad      = None
        self._load_generation = 0
        self._ring_buffer     = np.array([], dtype=np.float32)
        self._ring_lock       = threading.Lock()
        self._chunk_buffer    = []
        self._chunk_samples   = 0
        self._set_model()

        if VAD_MODULE_OK:
            self._vad = STTVad(
                silence_duration_ms=800,
                min_speech_duration_ms=500,
                max_speech_duration_s=15.0,
            )
            LOG_MSG.info("VAD active: %s", self._vad.backend_name)
        else:
            self._vad = None
            LOG_MSG.warning("VAD not available")

        self._queue_cond      = threading.Condition()
        self._jobs            = deque()
        self._process_thread  = None
        self._stop_processing = False
        self._draining        = False
        self._use_partial_results = False

    def __del__(self):
        LOG_MSG.info("Whisper __del__")
        self._stop_processing = True
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
        super().__del__()

    def destroy(self):
        with self._queue_cond:
            self._stop_processing = True
            self._jobs.clear()
            self._queue_cond.notify_all()

        self._load_generation += 1

        thread_alive = False
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
            thread_alive = self._process_thread.is_alive()
            self._process_thread = None

        self._current_locale.disconnect(self._locale_id)
        self._locale_id = 0

        if self._model_id != 0:
            self._model.disconnect(self._model_id)
            self._model_id = 0

        self._appsink = None

        if thread_alive == False:
            self._whisper = None

        self._vad     = None
        LOG_MSG.info("Whisper.destroy() called")
        super().destroy()

    def _build_lang_code(self):
        if not self._current_locale or not self._current_locale.locale:
            return None
        code = self._current_locale.locale[:2].lower()
        return code if code.isalpha() else None

    def _load_whisper_model(self, model_path):
        """Load Whisper model using pywhispercpp"""
        if not WHISPER_AVAILABLE:
            LOG_MSG.error("pywhispercpp not available")
            return None

        try:
            LOG_MSG.info("Loading Whisper model: %s", model_path)
            lang_code = self._build_lang_code()

            kwargs = dict(
                print_realtime=False,
                print_progress=False,
                single_segment=True,
                no_context=True,
                audio_ctx=AUDIO_CTX,
                temperature_inc=TEMPERATURE_INC,
                max_tokens=MAX_TOKENS,
                n_threads=N_THREADS,
                no_speech_thold=0.6,
                logprob_thold=-1.5,
                entropy_thold=2.8,
                suppress_blank=True,
            )

            if lang_code:
                kwargs["language"] = lang_code

            return Model(model_path, **kwargs)

        except Exception as e:
            LOG_MSG.error("Failed to load Whisper model: %s", e)
            return None

    def _load_model_thread(self, model_path, restore_state, generation):
    # Load the model in the worker and schedule completion on the main loop
    # @model_path: Path to the selected model
    # @restore_state: Pipeline state before loading
    # @generation: ID of the model load request
        whisper = self._load_whisper_model(model_path)
        GLib.idle_add(self._model_load_done_cb,
                      whisper, restore_state, generation)

    def _model_load_done_cb(self, whisper, restore_state, generation):
    # Apply the model load result if the request is still current
    # @whisper: Loaded Whisper model, or None on failure
        if self.pipeline is None or generation != self._load_generation:
            LOG_MSG.debug("discarding stale model load")
            return False

        self._whisper = whisper
        if whisper is None:
            return False

        # Serialize the VAD reset with incoming audio
        with self._ring_lock:
            if self._vad is not None:
                self._vad.reset()

        self.restore_state_after_load(restore_state)

        self.emit("model-changed")
        return False

    def _set_model_path(self):
        if self._model is None or self._model.available() is False:
            LOG_MSG.info("model path does not exist (%s - %s)",
                        self._model.get_name() if self._model else "None",
                        self._model.get_path() if self._model else "None")
            self._load_generation += 1
            self._whisper = None
            self.emit("model-changed")
            return

        new_model_path = self._model.get_path()
        LOG_MSG.debug("model ready %s", new_model_path)

        ret, state, pending = self.pipeline.get_state(0)
        if state >= Gst.State.READY:
            self.pipeline.set_state(Gst.State.READY)

        self._load_generation += 1
        self._whisper = None
        threading.Thread(target=self._load_model_thread,
                         args=(new_model_path, state, self._load_generation),
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

        self._model = STTWhisperModel(locale_str=self._current_locale.locale)
        self._model_id = self._model.connect("changed", self._model_changed)
        self._set_model_path()

    def _locale_changed(self, locale):
        self._set_model()

    def _on_new_sample(self, appsink):
        if self._draining or self._stop_processing:
            sample = appsink.emit("pull-sample")
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

        # Serialize audio buffering with session changes
        with self._ring_lock:
            if session != self._session_id:
                LOG_MSG.debug("focus changed while this audio was read, "
                              "dropping it")
                return Gst.FlowReturn.OK

            if self._vad is not None:
                segments = self._vad.process(audio_float)
                snapshot = None
            else:
                segments = []
                snapshot = None
                self._chunk_buffer.append(audio_data)
                self._chunk_samples += len(audio_data)
                window_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
                if self._chunk_samples >= window_samples:
                    self._flush_chunks_to_ring_locked()
                    snapshot = self._ring_snapshot_locked()

        now = time.monotonic()
        for segment in segments:
            self._enqueue_for_transcription(
                segment.audio, 'vad', session,
                now - segment.trailing_silence_samples / SAMPLE_RATE)

        if snapshot is not None:
            self._enqueue_for_transcription(snapshot, 'ring', session, now)

        return Gst.FlowReturn.OK

    def _flush_chunks_to_ring_locked(self):
        # Caller holds _ring_lock.
        if not self._chunk_buffer:
            return

        new_samples = np.concatenate(self._chunk_buffer).astype(np.float32) / 32768.0
        self._chunk_buffer.clear()
        self._chunk_samples = 0

        max_samples = int(WINDOW_SECONDS * SAMPLE_RATE)
        self._ring_buffer = np.concatenate([self._ring_buffer, new_samples])
        if len(self._ring_buffer) > max_samples:
            self._ring_buffer = self._ring_buffer[-max_samples:]

    def _ring_snapshot_locked(self):
        # Caller holds _ring_lock.
        if len(self._ring_buffer) < MIN_SAMPLES:
            return None

        return self._ring_buffer.copy()

    def _enqueue_for_transcription(self, audio_float32: np.ndarray, source: str,
                                   session: int, speech_end: float = None):
    # @speech_end: monotonic time the talker is estimated to have stopped
        if speech_end is None:
            speech_end = time.monotonic()

        job = (source, audio_float32, session, speech_end, time.monotonic())

        with self._queue_cond:
            if self._stop_processing:
                LOG_MSG.debug("backend is stopping, dropping %s job", source)
                return

            if source == 'ring':
                # Limit pending ring snapshots and retain each VAD utterance
                pending = [i for i, item in enumerate(self._jobs)
                           if item[0] == 'ring']
                if len(pending) >= MAX_RING_QUEUE_DEPTH:
                    del self._jobs[pending[0]]

            self._jobs.append(job)
            depth = len(self._jobs)

            # The appsink and main loop can both start the worker
            if self._process_thread is None or \
               not self._process_thread.is_alive():
                self._process_thread = threading.Thread(
                    target=self._process_worker, daemon=True)
                self._process_thread.start()

            self._queue_cond.notify()

        if depth > 2:
            LOG_MSG.warning("transcription backlog: %d jobs waiting, the next "
                            "utterance will be late", depth)

    def _next_job(self):
    # Wait for queued audio
    # Returns --> Oldest job, or None on timeout/shutdown
        with self._queue_cond:
            if self._stop_processing:
                return None
            if not self._jobs:
                self._queue_cond.wait(0.1)
            if not self._jobs or self._stop_processing:
                return None
            return self._jobs.popleft()

    def _process_worker(self):
        """Background worker to process audio"""
        while not self._stop_processing:
            job = self._next_job()
            if job is None:
                continue

            source, audio, session, speech_end, queued_at = job

            if self._whisper is None:
                continue

            if session != self._session_id:
                LOG_MSG.debug("dropping audio captured for session %i", session)
                continue

            try:
                started = time.monotonic()
                segments = self._whisper.transcribe(audio)
                done = time.monotonic()

                text_parts = []
                for segment in segments:
                    if not hasattr(segment, 'text'):
                        continue

                    segment_text = segment.text.strip()
                    if SPECIAL_PATTERN.match(segment_text):
                        continue

                    if segment_text:
                        text_parts.append(segment_text)

                text = ' '.join(text_parts).strip()

                LOG_MSG.info(
                    "Whisper %s: %.2f s audio | endpoint %.2f s + queue "
                    "%.2f s + inference %.2f s = %.2f s estimated speech end "
                    "to decode completion",
                    source, len(audio) / SAMPLE_RATE,
                    queued_at - speech_end, started - queued_at,
                    done - started, done - speech_end)
                LOG_MSG.debug("Whisper %s decoded: '%s'", source, text)

                if text:
                    GLib.idle_add(self._emit_text, text, session)

            except Exception as e:
                LOG_MSG.error("Whisper transcription error: %s", e, exc_info=True)

    def _emit_text(self, text, session):
        if session != self._session_id:
            LOG_MSG.debug("dropping result of session %i (current is %i)",
                          session, self._session_id)
            return False

        self.emit("text", text)
        return False

    def discard_pending(self):
    # Invalidate the session and clear buffered Whisper audio
        with self._ring_lock:
            super().discard_pending()

            self._ring_buffer = np.array([], dtype=np.float32)
            self._chunk_buffer.clear()
            self._chunk_samples = 0

            if self._vad is not None:
                self._vad.reset()

    def get_final_results(self):
        # Let the worker finish asynchronously so IBus can handle key events
        if self._whisper is None or self._stop_processing:
            LOG_MSG.debug("no model loaded, nothing to flush")
            return

        self._draining = True
        try:
            with self._ring_lock:
                session = self._session_id
                if self._vad is not None:
                    remaining = self._vad.flush()
                    snapshot = None
                else:
                    remaining = None
                    self._flush_chunks_to_ring_locked()
                    snapshot = self._ring_snapshot_locked()

            if remaining is not None:
                self._enqueue_for_transcription(
                    remaining.audio, 'vad', session,
                    time.monotonic()
                    - remaining.trailing_silence_samples / SAMPLE_RATE)

            if snapshot is not None:
                self._enqueue_for_transcription(snapshot, 'ring', session)
        finally:
            self._draining = False

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
        self.get_final_results()
        return super()._stop_real()
