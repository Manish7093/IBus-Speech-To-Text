import logging
import os
import numpy as np
from collections import deque
from typing import Optional

LOG_MSG = logging.getLogger()

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


class _SileroClassifier:
    _CHUNK_SAMPLES = 512

    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(model_path, sess_options=opts)
        input_names = {inp.name for inp in self._session.get_inputs()}
        self._v4 = 'h' in input_names
        if self._v4:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        else:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._sr = np.array(16000, dtype=np.int64)

    @property
    def chunk_samples(self) -> int:
        return self._CHUNK_SAMPLES

    def reset(self) -> None:
        if self._v4:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        else:
            self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def classify(self, chunk: np.ndarray) -> float:
        input_tensor = chunk.reshape(1, -1)
        if self._v4:
            out, self._h, self._c = self._session.run(
                None,
                {'input': input_tensor, 'sr': self._sr,
                 'h': self._h, 'c': self._c})
        else:
            out, self._state = self._session.run(
                None,
                {'input': input_tensor, 'state': self._state,
                 'sr': self._sr})
        return float(out[0][0])


class _EnergyClassifier:
    _CHUNK_SAMPLES = 320

    def __init__(self, freq_thold: float, low_energy_ratio: float,
                 rms_threshold: float, sample_rate: int):
        self._freq_thold       = freq_thold
        self._low_energy_ratio = low_energy_ratio
        self._rms_threshold    = rms_threshold
        self._sample_rate      = sample_rate

    @property
    def chunk_samples(self) -> int:
        return self._CHUNK_SAMPLES

    def reset(self) -> None:
        pass

    def classify(self, chunk: np.ndarray) -> float:
        if self._freq_thold > 0.0 and not self._spectral_gate(chunk):
            return 0.0
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        return 1.0 if rms >= self._rms_threshold else 0.0

    def _spectral_gate(self, chunk: np.ndarray) -> bool:
        n            = len(chunk)
        fft_mag      = np.abs(np.fft.rfft(chunk))
        freqs        = np.fft.rfftfreq(n, d=1.0 / self._sample_rate)
        energy_total = float(np.dot(fft_mag, fft_mag))
        if energy_total < 1e-12:
            return False
        low_mask   = freqs < self._freq_thold
        energy_low = float(np.dot(fft_mag[low_mask], fft_mag[low_mask]))
        return (energy_low / energy_total) < self._low_energy_ratio


_SILERO_MODEL_FILENAME = "silero_vad.onnx"


def _find_silero_model() -> Optional[str]:
    candidates = [
        os.environ.get("IBUS_STT_SILERO_VAD_MODEL"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "models", _SILERO_MODEL_FILENAME),
        os.path.join("/usr/share/ibus-stt/models", _SILERO_MODEL_FILENAME),
        os.path.join("/usr/local/share/ibus-stt/models", _SILERO_MODEL_FILENAME),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.realpath(path)
    return None


class STTVad:
    SAMPLE_RATE = 16000
    _ENERGY_RMS_THRESHOLD = 0.015

    def __init__(
        self,
        speech_threshold: float = 0.5,
        silence_duration_ms: int = 600,
        speech_pad_ms: int = 200,
        min_speech_duration_ms: int = 300,
        max_speech_duration_s: float = 30.0,
        freq_thold: float = 100.0,
        low_energy_ratio: float = 0.88,
    ):
        self._threshold = speech_threshold

        self._silence_samples    = int(silence_duration_ms    * self.SAMPLE_RATE / 1000)
        self._pad_samples        = int(speech_pad_ms          * self.SAMPLE_RATE / 1000)
        self._min_speech_samples = int(min_speech_duration_ms * self.SAMPLE_RATE / 1000)
        self._max_speech_samples = int(max_speech_duration_s  * self.SAMPLE_RATE)

        self._classifier = None
        if _ORT_AVAILABLE:
            model_path = _find_silero_model()
            if model_path:
                try:
                    self._classifier = _SileroClassifier(model_path)
                    LOG_MSG.info("Silero VAD loaded from %s", model_path)
                except Exception as e:
                    LOG_MSG.warning("Failed to load Silero VAD: %s", e)

        if self._classifier is None:
            self._classifier = _EnergyClassifier(
                freq_thold, low_energy_ratio,
                self._ENERGY_RMS_THRESHOLD, self.SAMPLE_RATE)
            if _ORT_AVAILABLE:
                LOG_MSG.info("Silero VAD model not found, using energy-based VAD")
            else:
                LOG_MSG.info("onnxruntime not installed, using energy-based VAD")

        self._chunk_size = self._classifier.chunk_samples

        pad_chunks = max(1, self._pad_samples // self._chunk_size)
        self._pre_buf: deque = deque(maxlen=pad_chunks)

        self._leftover        = np.empty(0, dtype=np.float32)
        self._speech_chunks   = []
        self._speech_samples  = 0
        self._in_speech       = False
        self._silence_counter = 0

        LOG_MSG.info(
            "STTVad ready -- backend=%s | chunk=%d samples | "
            "threshold=%.2f | silence=%dms | pad=%dms | min=%dms | max=%.1fs",
            self.backend, self._chunk_size, self._threshold,
            silence_duration_ms, speech_pad_ms,
            min_speech_duration_ms, max_speech_duration_s)

    @property
    def backend(self) -> str:
        return "silero" if isinstance(self._classifier, _SileroClassifier) else "energy"

    @property
    def backend_name(self) -> str:
        if isinstance(self._classifier, _SileroClassifier):
            return "Silero VAD (ONNX)"
        return "Energy-based VAD"

    def reset(self) -> None:
        self._classifier.reset()
        self._pre_buf.clear()
        self._leftover        = np.empty(0, dtype=np.float32)
        self._speech_chunks   = []
        self._speech_samples  = 0
        self._in_speech       = False
        self._silence_counter = 0

    def process(self, audio: np.ndarray) -> list:
        completed_segments = []

        if len(self._leftover):
            audio = np.concatenate([self._leftover, audio])
        self._leftover = np.empty(0, dtype=np.float32)

        offset = 0
        while offset + self._chunk_size <= len(audio):
            chunk   = audio[offset: offset + self._chunk_size]
            offset += self._chunk_size

            prob = self._classifier.classify(chunk)
            if prob >= self._threshold:
                self._on_speech_chunk(chunk, prob, completed_segments)
            else:
                self._on_silence_chunk(chunk, completed_segments)

        if offset < len(audio):
            self._leftover = audio[offset:].copy()

        return completed_segments

    def flush(self):
        segment = None
        if self._in_speech and self._speech_samples >= self._min_speech_samples:
            parts = list(self._speech_chunks)
            if len(self._leftover):
                parts.append(self._leftover)
            segment = np.concatenate(parts).astype(np.float32)

        self._speech_chunks   = []
        self._speech_samples  = 0
        self._silence_counter = 0
        self._in_speech       = False
        self._leftover        = np.empty(0, dtype=np.float32)
        return segment

    def _on_speech_chunk(self, chunk, confidence, out_segments):
        if not self._in_speech:
            LOG_MSG.debug("VAD: speech onset (confidence=%.3f)", confidence)
            self._in_speech       = True
            self._silence_counter = 0
            for pre_chunk in self._pre_buf:
                self._speech_chunks.append(pre_chunk)
                self._speech_samples += len(pre_chunk)

        self._speech_chunks.append(chunk)
        self._speech_samples  += len(chunk)
        self._silence_counter  = 0

        if self._speech_samples >= self._max_speech_samples:
            segment = np.concatenate(self._speech_chunks).astype(np.float32)
            out_segments.append(segment)
            self._speech_chunks  = []
            self._speech_samples = 0

    def _on_silence_chunk(self, chunk, out_segments):
        self._pre_buf.append(chunk)
        if not self._in_speech:
            return

        self._speech_chunks.append(chunk)
        self._speech_samples  += len(chunk)
        self._silence_counter += len(chunk)

        if self._silence_counter >= self._silence_samples:
            duration = self._speech_samples / self.SAMPLE_RATE
            if self._speech_samples >= self._min_speech_samples:
                segment = np.concatenate(self._speech_chunks).astype(np.float32)
                out_segments.append(segment)
            else:
                LOG_MSG.debug(
                    "VAD: discarding short segment (%.3f s < %.3f s min)",
                    duration,
                    self._min_speech_samples / self.SAMPLE_RATE)
            self._speech_chunks   = []
            self._speech_samples  = 0
            self._silence_counter = 0
            self._in_speech       = False
