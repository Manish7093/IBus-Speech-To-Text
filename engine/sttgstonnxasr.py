# vim:set et sts=4 sw=4:
#
# ibus-stt - Speech To Text engine for IBus
# Copyright (C) 2022 Philippe Rouquier <bonfire-app@wanadoo.fr>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import threading
import queue
import numpy as np

from gi.repository import Gst, GLib
from sttutils import *
from sttgstbase import STTGstBase
from sttcurrentlocale import stt_current_locale
from sttonnxasrmodel import STTOnnxAsrModel

LOG_MSG = logging.getLogger()

try:
    import onnx_asr
    ONNXASR_AVAILABLE = True
except ImportError:
    LOG_MSG.warning("onnx-asr not available. Install with: pip install onnx-asr")
    ONNXASR_AVAILABLE = False


class STTGstOnnxAsr(STTGstBase):
    __gtype_name__ = 'STTGstOnnxAsr'
    _pipeline_def = "pulsesrc blocksize=3200 buffer-time=9223372036854775807 ! " \
                    "audio/x-raw,format=S16LE,rate=16000,channels=1 ! " \
                    "webrtcdsp noise-suppression-level=3 echo-cancel=false ! " \
                    "queue ! " \
                    "appsink name=OnnxAsrSink emit-signals=true sync=false"

    _pipeline_def_alt = "pulsesrc blocksize=3200 buffer-time=9223372036854775807 ! " \
                        "audio/x-raw,format=S16LE,rate=16000,channels=1 ! " \
                        "queue ! " \
                        "appsink name=OnnxAsrSink emit-signals=true sync=false"

    def __init__(self, current_locale=None):
        plugin = Gst.Registry.get().find_plugin("webrtcdsp")
        if plugin is not None:
            super().__init__(pipeline_definition=STTGstOnnxAsr._pipeline_def)
            LOG_MSG.debug("using Webrtcdsp plugin")
        else:
            super().__init__(pipeline_definition=STTGstOnnxAsr._pipeline_def_alt)
            LOG_MSG.debug("not using Webrtcdsp plugin")

        if self.pipeline is None:
            LOG_MSG.error("pipeline was not created")
            return

        self._appsink = self.pipeline.get_by_name("OnnxAsrSink")
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
        self._model = None
        self._recognizer = None
        self._set_model()

        self._audio_buffer = []
        self._buffer_duration = 0.0
        self._max_buffer_duration = 6.0
        self._min_buffer_duration = 2.0
        self._sample_rate = 16000
        self._processing = False
        self._process_queue = queue.Queue()
        self._process_thread = None
        self._stop_processing = False

        self._use_partial_results = False
        self._audio_lock = threading.Lock()

    def __del__(self):
        LOG_MSG.info("OnnxAsr __del__")
        self._stop_processing = True
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None
        super().__del__()

    def destroy(self):
        self._stop_processing = True
        if self._process_thread is not None:
            self._process_thread.join(timeout=2.0)
            self._process_thread = None

        self._current_locale.disconnect(self._locale_id)
        self._locale_id = 0

        if self._model_id != 0:
            self._model.disconnect(self._model_id)
            self._model_id = 0

        self._appsink = None
        self._recognizer = None

        LOG_MSG.info("OnnxAsr.destroy() called")
        super().destroy()

    def _load_onnxasr_model(self, model_name):
        """Load onnx-asr model."""
        if not ONNXASR_AVAILABLE:
            LOG_MSG.error("onnx-asr not available")
            return False

        try:
            LOG_MSG.info("Loading onnx-asr model: %s", model_name)
            self._recognizer = onnx_asr.load_model(model_name)
            LOG_MSG.info("onnx-asr model loaded successfully")
            return True

        except Exception as e:
            LOG_MSG.error("Failed to load onnx-asr model: %s", e)
            self._recognizer = None
            return False

    def _set_model_path(self):
        if self._model is None or self._model.available() is False:
            LOG_MSG.info("model not available (%s - %s)",
                        self._model.get_name() if self._model else "None",
                        self._model.get_path() if self._model else "None")
            self._recognizer = None
            self.emit("model-changed")
            return

        model_name = self._model.get_path()
        LOG_MSG.debug("model ready %s", model_name)

        ret, state, pending = self.pipeline.get_state(0)
        if state >= Gst.State.READY:
            self.pipeline.set_state(Gst.State.READY)

        success = self._load_onnxasr_model(model_name)

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

        self._model = STTOnnxAsrModel(locale_str=self._current_locale.locale)
        self._model_id = self._model.connect("changed", self._model_changed)
        self._set_model_path()

    def _locale_changed(self, locale):
        self._set_model()

    def _on_new_sample(self, appsink):
        """Callback when new audio sample arrives."""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.OK

        audio_data = np.frombuffer(map_info.data, dtype=np.int16)
        buf.unmap(map_info)

        with self._audio_lock:
            self._audio_buffer.append(audio_data)
            self._buffer_duration += len(audio_data) / self._sample_rate
            trigger_process = self._buffer_duration >= self._max_buffer_duration

        if trigger_process:
            self._process_audio_buffer()

        return Gst.FlowReturn.OK

    def _process_audio_buffer(self, force=False):
        """Process accumulated audio buffer."""
        with self._audio_lock:
            if len(self._audio_buffer) == 0:
                return

            if not force and self._buffer_duration < self._min_buffer_duration:
                LOG_MSG.debug("Buffer too short (%.2fs), waiting for more audio", self._buffer_duration)
                return

            if self._recognizer is None:
                LOG_MSG.warning("onnx-asr model not loaded")
                self._audio_buffer.clear()
                self._buffer_duration = 0.0
                return

            audio = np.concatenate(self._audio_buffer)
            self._audio_buffer.clear()
            self._buffer_duration = 0.0

            LOG_MSG.debug("Processing audio buffer: %d samples (%.2f seconds)",
                         len(audio), len(audio) / self._sample_rate)

            audio_float = audio.astype(np.float32) / 32768.0

            self._process_queue.put(audio_float)

            if self._process_thread is None or not self._process_thread.is_alive():
                self._process_thread = threading.Thread(target=self._process_worker, daemon=True)
                self._process_thread.start()

    def _process_worker(self):
        """Background worker to process audio."""
        while not self._stop_processing:
            try:
                audio = self._process_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            recognizer = self._recognizer
            if recognizer is None:
                self._process_queue.task_done()
                continue

            try:
                LOG_MSG.debug("Starting transcription of %d samples", len(audio))

                text = recognizer.recognize(audio, sample_rate=self._sample_rate)

                if isinstance(text, str):
                    text = text.strip()

                if text:
                    LOG_MSG.info("onnx-asr transcription result: '%s'", text)
                    GLib.idle_add(self._emit_text, text)
                else:
                    LOG_MSG.debug("No text transcribed from audio")

            except Exception as e:
                LOG_MSG.error("onnx-asr transcription error: %s", e, exc_info=True)

            self._process_queue.task_done()

    def _emit_text(self, text):
        self.emit("text", text)
        return False

    def get_final_results(self):
        self._process_audio_buffer(force=True)
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
        self.get_final_results()
        return super()._stop_real()
