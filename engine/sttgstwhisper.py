import logging
import threading
import queue
import numpy as np
from pathlib import Path

from gi.repository import Gst, GLib

from sttutils import *
from sttgstbase import STTGstBase
from sttcurrentlocale import stt_current_locale
from sttwhispermodel import STTWhisperModel

LOG_MSG = logging.getLogger()

# Try to import pywhispercpp Python bindings
try:
    from pywhispercpp.model import Model
    WHISPER_AVAILABLE = True
except ImportError:
    LOG_MSG.warning("pywhispercpp not available. Install with: pip install pywhispercpp")
    WHISPER_AVAILABLE = False


class STTGstWhisper(STTGstBase):
    __gtype_name__ = 'STTGstWhisper'

    # Pipeline captures audio and sends to appsink for processing
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

        # Connect to new-sample signal
        self._appsink.connect("new-sample", self._on_new_sample)

        if current_locale is None:
            self._current_locale = stt_current_locale()
        else:
            self._current_locale = current_locale

        self._locale_id = self._current_locale.connect("changed", self._locale_changed)

        self._model_id = 0
        self._model = None
        self._whisper = None
        self._set_model()

        # Audio buffering for batch processing
        self._audio_buffer = []
        self._buffer_duration = 0.0
        self._max_buffer_duration = 6.0  # Process every 2 seconds (reduced from 3)
        self._min_buffer_duration = 2.0  # Minimum 1 second before processing
        self._sample_rate = 16000
        self._processing = False
        self._process_queue = queue.Queue()
        self._process_thread = None
        self._stop_processing = False

        # Partial results support
        self._use_partial_results = False

    def __del__(self):
        LOG_MSG.info("Whisper __del__")
        self._stop_processing = True
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
        super().__del__()

    def destroy(self):
        self._stop_processing = True
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)

        self._current_locale.disconnect(self._locale_id)
        self._locale_id = 0

        if self._model_id != 0:
            self._model.disconnect(self._model_id)
            self._model_id = 0

        self._appsink = None
        self._whisper = None

        LOG_MSG.info("Whisper.destroy() called")
        super().destroy()

    def _load_whisper_model(self, model_path):
        """Load Whisper model using pywhispercpp"""
        if not WHISPER_AVAILABLE:
            LOG_MSG.error("pywhispercpp not available")
            return False

        try:
            LOG_MSG.info("Loading Whisper model: %s", model_path)
            
            # Determine language from locale
            lang_code = None
            if self._current_locale and self._current_locale.locale:
                lang_code = self._current_locale.locale[:2]
            
            # Create Model instance with language parameter
            # Model will auto-download if path is just a model name
            if lang_code and lang_code != 'multilingual':
                self._whisper = Model(model_path, language=lang_code, 
                                     print_realtime=False, print_progress=False)
            else:
                # Auto-detect language for multilingual models
                self._whisper = Model(model_path, 
                                     print_realtime=False, print_progress=False)

            LOG_MSG.info("Whisper model loaded successfully")
            return True
            
        except Exception as e:
            LOG_MSG.error("Failed to load Whisper model: %s", e)
            self._whisper = None
            return False

    def _set_model_path(self):
        if self._model is None or self._model.available() is False:
            LOG_MSG.info("model path does not exist (%s - %s)",
                        self._model.get_name() if self._model else "None",
                        self._model.get_path() if self._model else "None")
            self._whisper = None
            self.emit("model-changed")
            return

        new_model_path = self._model.get_path()
        LOG_MSG.debug("model ready %s", new_model_path)

        # Load the model
        ret, state, pending = self.pipeline.get_state(0)
        if state >= Gst.State.READY:
            self.pipeline.set_state(Gst.State.READY)

        success = self._load_whisper_model(new_model_path)

        if state >= Gst.State.READY:
            self.pipeline.set_state(state)

        if success:
            self.emit("model-changed")

    def _model_changed(self, model):
        self._set_model_path()

    def _set_model(self):
        if self._model is not None and \
           self._model.get_locale() == self._current_locale.locale:
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
        """Callback when new audio sample arrives"""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        # Convert to numpy array (16-bit PCM)
        audio_data = np.frombuffer(map_info.data, dtype=np.int16)
        buf.unmap(map_info)

        # Add to buffer
        self._audio_buffer.append(audio_data)
        self._buffer_duration += len(audio_data) / self._sample_rate

        # Process if buffer is full
        if self._buffer_duration >= self._max_buffer_duration:
            self._process_audio_buffer()

        return Gst.FlowReturn.OK

    def _process_audio_buffer(self):
        """Process accumulated audio buffer"""
        if len(self._audio_buffer) == 0:
            return

        # Don't process if buffer is too short
        if self._buffer_duration < self._min_buffer_duration:
            LOG_MSG.debug("Buffer too short (%.2fs), waiting for more audio", self._buffer_duration)
            return

        if self._whisper is None:
            LOG_MSG.warning("Whisper model not loaded")
            self._audio_buffer.clear()
            self._buffer_duration = 0.0
            return

        # Concatenate all audio chunks
        audio = np.concatenate(self._audio_buffer)
        self._audio_buffer.clear()
        self._buffer_duration = 0.0

        LOG_MSG.debug("Processing audio buffer: %d samples (%.2f seconds)", 
                     len(audio), len(audio) / self._sample_rate)

        # Convert to float32 [-1.0, 1.0] as required by Whisper
        audio_float = audio.astype(np.float32) / 32768.0

        # Queue for processing in background thread
        self._process_queue.put(audio_float)

        # Start processing thread if not running
        if self._process_thread is None or not self._process_thread.is_alive():
            self._process_thread = threading.Thread(target=self._process_worker, daemon=True)
            self._process_thread.start()

    def _process_worker(self):
        """Background worker to process audio"""
        while not self._stop_processing:
            try:
                audio = self._process_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._whisper is None:
                self._process_queue.task_done()
                continue

            try:
                LOG_MSG.debug("Starting transcription of %d samples", len(audio))
                
                # Transcribe audio using pywhispercpp
                # The Model.transcribe() method returns a generator of segments
                segments = self._whisper.transcribe(audio)
                
                # Collect all text from segments
                text_parts = []
                for segment in segments:
                    # segment.text contains the transcribed text
                    if hasattr(segment, 'text'):
                        segment_text = segment.text.strip()
                        if segment_text:
                            text_parts.append(segment_text)
                            LOG_MSG.debug("Segment text: '%s'", segment_text)
                
                text = ' '.join(text_parts).strip()

                if text:
                    LOG_MSG.info("Whisper transcription result: '%s'", text)
                    # Emit on main thread
                    GLib.idle_add(self._emit_text, text)
                else:
                    LOG_MSG.debug("No text transcribed from audio")

            except Exception as e:
                LOG_MSG.error("Whisper transcription error: %s", e, exc_info=True)

            self._process_queue.task_done()

    def _emit_text(self, text):
        """Emit text signal on main thread"""
        # Always emit as final text to ensure it gets committed
        self.emit("text", text)
        return False  # Don't repeat

    def get_final_results(self):
        """Process any remaining audio in buffer"""
        if len(self._audio_buffer) > 0:
            self._process_audio_buffer()

        # Wait for processing to complete
        self._process_queue.join()

    def get_results(self):
        """Get current results"""
        # With batch processing, we don't have intermediate results
        pass

    def set_use_partial_results(self, active):
        """Enable/disable partial results"""
        self._use_partial_results = active

    def set_alternatives_num(self, num):
        """Set number of alternatives (not supported by basic Whisper)"""
        # Whisper doesn't provide alternatives by default
        # Could be implemented with beam search or multiple passes
        pass

    def has_model(self):
        if self._model is None or self._model.available() is False:
            return False

        return super().has_model()

    def _stop_real(self):
        """Override to process remaining audio before stopping"""
        self.get_final_results()
        return super()._stop_real()
