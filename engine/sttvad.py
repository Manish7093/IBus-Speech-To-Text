import logging
import numpy as np
from collections import deque

LOG_MSG = logging.getLogger()

class STTVad:
    _ENERGY_CHUNK_SAMPLES = 320
    SAMPLE_RATE           = 16000

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
        self._threshold        = speech_threshold
        self._freq_thold       = freq_thold
        self._low_energy_ratio = low_energy_ratio

        self._silence_samples    = int(silence_duration_ms    * self.SAMPLE_RATE / 1000)
        self._pad_samples        = int(speech_pad_ms          * self.SAMPLE_RATE / 1000)
        self._min_speech_samples = int(min_speech_duration_ms * self.SAMPLE_RATE / 1000)
        self._max_speech_samples = int(max_speech_duration_s  * self.SAMPLE_RATE)

        self._chunk_size = self._ENERGY_CHUNK_SAMPLES

        pad_chunks = max(1, self._pad_samples // self._chunk_size)
        self._pre_buf: deque = deque(maxlen=pad_chunks)

        self._leftover        = np.empty(0, dtype=np.float32)
        self._speech_chunks   = []
        self._speech_samples  = 0
        self._in_speech       = False
        self._silence_counter = 0

        LOG_MSG.info(
            "STTVad ready -- backend=energy | chunk=%d samples | "
            "silence=%dms | pad=%dms | min=%dms | max=%.1fs | "
            "freq_thold=%.0fHz | low_energy_ratio=%.2f",
            self._chunk_size,
            silence_duration_ms, speech_pad_ms,
            min_speech_duration_ms, max_speech_duration_s,
            self._freq_thold, self._low_energy_ratio)

    @property
    def backend(self) -> str:
        return "energy"

    @property
    def backend_name(self) -> str:
        return "Energy-based VAD"

    def reset(self) -> None:
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

            if self._freq_thold > 0.0 and not self._spectral_gate(chunk):
                self._on_silence_chunk(chunk, completed_segments)
                continue

            is_speech, confidence = self._energy_classify(chunk)
            if is_speech:
                self._on_speech_chunk(chunk, confidence, completed_segments)
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

    def _spectral_gate(self, chunk: np.ndarray) -> bool:
        n            = len(chunk)
        fft_mag      = np.abs(np.fft.rfft(chunk))
        freqs        = np.fft.rfftfreq(n, d=1.0 / self.SAMPLE_RATE)
        energy_total = float(np.dot(fft_mag, fft_mag))
        if energy_total < 1e-12:
            return False   # silent chunk → not speech
        low_mask   = freqs < self._freq_thold
        energy_low = float(np.dot(fft_mag[low_mask], fft_mag[low_mask]))
        return (energy_low / energy_total) < self._low_energy_ratio

    @staticmethod
    def _energy_classify(chunk: np.ndarray) -> tuple:
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        return rms >= 0.015, rms

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
