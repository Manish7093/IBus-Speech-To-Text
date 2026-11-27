import logging
import numpy as np
from collections import deque, namedtuple

LOG_MSG = logging.getLogger()

STTVadSegment = namedtuple('STTVadSegment', 'audio trailing_silence_samples')

_ENERGY_THRESHOLD = 0.015
_SPEECH_BAND_HZ   = 3000.0


class STTVad:
    _ENERGY_CHUNK_SAMPLES = 320
    SAMPLE_RATE           = 16000

    def __init__(
        self,
        silence_duration_ms: int = 600,
        speech_pad_ms: int = 200,
        min_speech_duration_ms: int = 300,
        max_speech_duration_s: float = 30.0,
        freq_thold: float = 100.0,
        onset_duration_ms: int = 60,
        min_voiced_duration_ms: int = 100,
        min_voiced_ratio: float = 0.10,
        min_speech_band_ratio: float = 0.45,
    ):
        self._freq_thold       = freq_thold
        self._min_band_ratio   = min_speech_band_ratio
        self._min_voiced_ratio = min_voiced_ratio

        self._silence_samples    = int(silence_duration_ms    * self.SAMPLE_RATE / 1000)
        self._pad_samples        = int(speech_pad_ms          * self.SAMPLE_RATE / 1000)
        self._min_speech_samples = int(min_speech_duration_ms * self.SAMPLE_RATE / 1000)
        self._max_speech_samples = int(max_speech_duration_s  * self.SAMPLE_RATE)
        self._min_voiced_samples = int(min_voiced_duration_ms * self.SAMPLE_RATE / 1000)

        self._chunk_size = self._ENERGY_CHUNK_SAMPLES
        chunk_ms = self._chunk_size * 1000 // self.SAMPLE_RATE

        self._onset_chunks = max(1, int(onset_duration_ms) // chunk_ms)

        self._pre_buf: deque = deque(
            maxlen=max(1, self._pad_samples // self._chunk_size))
        self.reset()

        LOG_MSG.info("Energy VAD ready: %d ms silence, %d ms onset, voiced "
                     "minimum %d ms and %.0f%% of the active span, speech "
                     "band %.0f-%.0f Hz >= %.2f",
                     silence_duration_ms, self._onset_chunks * chunk_ms,
                     min_voiced_duration_ms, min_voiced_ratio * 100,
                     freq_thold, _SPEECH_BAND_HZ, min_speech_band_ratio)

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
        self._voiced_samples  = 0
        self._run             = 0
        self._in_speech       = False
        self._silence_counter = 0

    def process(self, audio: np.ndarray) -> list:
    # Return segments completed by this batch of audio
        completed_segments = []

        if len(self._leftover):
            audio = np.concatenate([self._leftover, audio])
        self._leftover = np.empty(0, dtype=np.float32)

        offset = 0
        while offset + self._chunk_size <= len(audio):
            self._feed(audio[offset: offset + self._chunk_size],
                       completed_segments)
            offset += self._chunk_size

        if offset < len(audio):
            self._leftover = audio[offset:].copy()

        return completed_segments

    def flush(self):
    # Reset the detector and return the last accepted segment, or None
        segment = None
        if self._in_speech:
            if len(self._leftover):
                self._speech_chunks.append(self._leftover)
                self._speech_samples += len(self._leftover)
            segment = self._finish_segment('flush')

        self.reset()
        return segment

    def _is_speech_candidate(self, chunk: np.ndarray) -> bool:
    # @chunk: one 20 ms mono chunk
    # Returns--> True if it has the energy and the spectral shape of speech
        if float(np.sqrt(np.mean(chunk ** 2))) < _ENERGY_THRESHOLD:
            return False

        fft_mag = np.abs(np.fft.rfft(chunk))
        freqs   = np.fft.rfftfreq(len(chunk), d=1.0 / self.SAMPLE_RATE)
        total   = float(np.dot(fft_mag, fft_mag))
        if total < 1e-12:
            return False

        band    = (freqs >= self._freq_thold) & (freqs < _SPEECH_BAND_HZ)
        in_band = float(np.dot(fft_mag[band], fft_mag[band]))
        return (in_band / total) >= self._min_band_ratio

    def _feed(self, chunk, out_segments):
    # Advance the detector by one chunk, appending any segment it completes
        candidate = self._is_speech_candidate(chunk)
        self._run = self._run + 1 if candidate else 0
        voiced    = self._run >= self._onset_chunks

        if voiced and not self._in_speech:
            LOG_MSG.debug("VAD: speech onset")
            self._in_speech       = True
            self._silence_counter = 0
            # Count the onset run as voiced and keep earlier audio as padding
            lead_in = len(self._pre_buf) - (self._onset_chunks - 1)
            for index, (pad_chunk, pad_candidate) in enumerate(self._pre_buf):
                self._speech_chunks.append(pad_chunk)
                self._speech_samples += len(pad_chunk)
                if pad_candidate and index >= lead_in:
                    self._voiced_samples += len(pad_chunk)
            self._pre_buf.clear()

        if not self._in_speech:
            self._pre_buf.append((chunk, candidate))
            return

        self._speech_chunks.append(chunk)
        self._speech_samples += len(chunk)

        if voiced:
            self._voiced_samples += len(chunk)
            self._silence_counter = 0
        else:
            self._silence_counter += len(chunk)
            if self._silence_counter >= self._silence_samples:
                self._in_speech = False

        if not self._in_speech:
            reason = 'silence'
        elif self._speech_samples >= self._max_speech_samples:
            reason = 'max'
        else:
            return

        segment = self._finish_segment(reason)
        if segment is not None:
            out_segments.append(segment)

    def _finish_segment(self, reason):
    # Close the current segment and clear its sample counts
    # @reason: End condition ('silence', 'max' or 'flush')
    # Returns--> Accepted segment, or None if it fails the speech thresholds
        samples, voiced = self._speech_samples, self._voiced_samples
        # Exclude trailing silence when measuring the voiced fraction
        active = max(samples - self._silence_counter, 1)

        segment = None
        if samples >= self._min_speech_samples and \
           voiced >= self._min_voiced_samples and \
           voiced / active >= self._min_voiced_ratio:
            segment = STTVadSegment(
                np.concatenate(self._speech_chunks).astype(np.float32),
                self._silence_counter)

        LOG_MSG.debug("VAD %s segment: %.2f s with %.0f ms voiced in a "
                      "%.2f s active span (%.0f%%) -> %s",
                      reason, samples / self.SAMPLE_RATE,
                      voiced * 1000 / self.SAMPLE_RATE,
                      active / self.SAMPLE_RATE, voiced * 100 / active,
                      "decode" if segment is not None else "dropped")

        self._speech_chunks  = []
        self._speech_samples = 0
        self._voiced_samples = 0
        return segment
