import math
import struct
from typing import Tuple, List
import brotli
import numpy as np

class PSCoder:
    """Analyze and apply stereo characteristics (IID, IPD, IC) across frequency bands.

    Uses logarithmic frequency scaling for perceptually uniform frequency analysis.
    """

    def __init__(
            self,
            sample_rate: int,
            min_freq: float = 20.0,
            max_freq: float = 20000.0,
            freq_points: int = 32,
            floor_db: float = -75.0,
            use_grouping: bool = False,
            log_scale: bool = True
    ):
        """Initialize the stereo audio analyzer.

        Args:
            sample_rate: Audio sample rate in Hz
            min_freq: Minimum frequency to analyze in Hz
            max_freq: Maximum frequency to analyze in Hz
            freq_points: Number of frequency points to sample
            floor_db: Noise floor in dB (signals below this are ignored)
            use_grouping: If True, average over frequency bands
            log_scale: If True, use logarithmic frequency spacing (default)
        """
        self.sample_rate = sample_rate
        self.floor_db = floor_db
        self.use_grouping = use_grouping
        self.log_scale = log_scale

        # Pre-calculate target frequencies
        self.set_freq(min_freq, max_freq, freq_points)

    def set_freq(self, min_freq: float, max_freq: float, freq_points: int):
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_points = freq_points

        if self.log_scale:
            self.target_freqs = np.logspace(
                np.log10(min_freq),
                np.log10(max_freq),
                freq_points
            )
        else:
            self.target_freqs = np.linspace(min_freq, max_freq, freq_points)


    def analyze(
            self,
            audio_data: np.ndarray,
            infloat=False
    ) -> List[Tuple[float, float, float, float]]:
        """Analyze stereo audio and extract IID (pan), IPD, and IC per frequency band.

        Args:
            audio_data: Stereo audio data with shape [samples, 2]

        Returns:
            List of tuples: [(freq, pan, ipd, ic), ...]
            - freq: frequency in Hz
            - pan: Inter-aural Intensity Difference (IID) as pan value [-1, 1]
            - ipd: Inter-aural Phase Difference in radians [-π, π]
            - ic: Inter-channel Coherence [0, 1]
        """
        if audio_data.ndim != 2 or audio_data.shape[1] != 2:
            raise ValueError("Audio must be stereo (shape: [samples, 2])")

        # Normalize int16 to float
        if infloat:
            audio_float = audio_data
        else:
            audio_float = audio_data.astype(np.float32) / 32768.0

        # Separate channels
        left = audio_float[:, 0]
        right = audio_float[:, 1]

        # Calculate FFT for both channels
        n_fft = len(left)
        left_fft = np.fft.rfft(left)
        right_fft = np.fft.rfft(right)

        # Get frequency bins
        freqs = np.fft.rfftfreq(n_fft, 1 / self.sample_rate)

        # Calculate magnitude for each channel
        left_mag = np.abs(left_fft)
        right_mag = np.abs(right_fft)

        # Calculate combined magnitude for floor detection
        combined_mag = (left_mag + right_mag) / 2

        # Convert floor from dB to linear scale
        floor_linear = 10 ** (self.floor_db / 20.0)

        # Calculate bandwidth for grouping
        if self.use_grouping and self.freq_points > 1:
            if self.log_scale:
                # For log scale, bandwidth is proportional to frequency
                bandwidths = self._calculate_log_bandwidths()
            else:
                bandwidth = (self.max_freq - self.min_freq) / self.freq_points
                bandwidths = [bandwidth] * self.freq_points
        else:
            bandwidths = [0] * self.freq_points

        # Analyze each frequency band
        results = []
        for target_freq, bandwidth in zip(self.target_freqs, bandwidths):
            if self.use_grouping and bandwidth > 0:
                freq_low = target_freq - bandwidth / 2
                freq_high = target_freq + bandwidth / 2
                mask = (freqs >= freq_low) & (freqs <= freq_high)

                if np.any(mask):
                    pan, ipd, ic = self._analyze_band(
                        left_fft[mask], right_fft[mask],
                        left_mag[mask], right_mag[mask],
                        combined_mag[mask], floor_linear
                    )
                else:
                    pan, ipd, ic = 0.0, 0.0, 0.0
            else:
                idx = np.argmin(np.abs(freqs - target_freq))
                pan, ipd, ic = self._analyze_bin(
                    left_fft[idx], right_fft[idx],
                    left_mag[idx], right_mag[idx],
                    combined_mag[idx], floor_linear
                )

            results.append((float(target_freq), float(pan), float(ipd), float(ic)))

        return results

    def apply(
            self,
            mono_audio: np.ndarray,
            pan_values: List[float],
            ipd_values: List[float],
            ic_values: List[bool]
    ) -> np.ndarray:
        """Apply frequency-dependent IID (pan), IPD (phase), and IC (coherence) to mono audio.

        Args:
            mono_audio: Mono audio data
            pan_values: Pan values for each frequency point [-1, 1]
            ipd_values: Phase difference values for each frequency point [radians]
            ic_values: Coherence flags for each frequency point [bool]

        Returns:
            Stereo audio data with shape [samples, 2]
        """
        # Normalize if int16
        if mono_audio.dtype == np.int16:
            mono_float = mono_audio.astype(np.float32) / 32768.0
            return_int16 = True
        else:
            mono_float = mono_audio.astype(np.float32)
            return_int16 = False

        # FFT
        mono_fft = np.fft.rfft(mono_float)
        freqs = np.fft.rfftfreq(len(mono_float), 1 / self.sample_rate)

        # Calculate bandwidth for grouping
        if self.use_grouping and self.freq_points > 1:
            if self.log_scale:
                bandwidths = self._calculate_log_bandwidths()
            else:
                bandwidth = (self.max_freq - self.min_freq) / self.freq_points
                bandwidths = [bandwidth] * self.freq_points
        else:
            bandwidths = [0] * self.freq_points

        # Initialize gain and phase shift arrays
        left_gain = np.ones(len(freqs), dtype=np.float32)
        right_gain = np.ones(len(freqs), dtype=np.float32)
        right_phase_shift = np.ones(len(freqs), dtype=np.complex64)

        # Apply stereo parameters to each frequency band
        for freq, pan, ipd, ic, bandwidth in zip(
                self.target_freqs, pan_values, ipd_values, ic_values, bandwidths
        ):
            # Constant power panning
            angle = (-pan + 1.0) * np.pi / 4.0
            left_g = np.cos(angle)
            right_g = np.sin(angle)

            # If IC is False, blend toward mono
            if not ic:
                left_g = right_g = (left_g + right_g) / 2.0
                ipd = 0.0  # remove phase difference

            if self.use_grouping and bandwidth > 0:
                freq_low = freq - bandwidth / 2
                freq_high = freq + bandwidth / 2
                mask = (freqs >= freq_low) & (freqs <= freq_high)
                left_gain[mask] = left_g
                right_gain[mask] = right_g
                right_phase_shift[mask] *= np.exp(1j * -ipd)
            else:
                idx = np.argmin(np.abs(freqs - freq))
                left_gain[idx] = left_g
                right_gain[idx] = right_g
                right_phase_shift[idx] *= np.exp(1j * -ipd)

        # Apply gains and phase
        left_fft = mono_fft * left_gain
        right_fft = mono_fft * right_gain * right_phase_shift

        left = np.fft.irfft(left_fft, n=len(mono_float))
        right = np.fft.irfft(right_fft, n=len(mono_float))

        stereo = np.column_stack([left, right])

        if return_int16:
            stereo = np.clip(stereo * 32767.0, -32768, 32767).astype(np.int16)

        return stereo

    def _calculate_log_bandwidths(self) -> List[float]:
        """Calculate bandwidths for logarithmic frequency spacing."""
        bandwidths = []
        log_freqs = np.log10(self.target_freqs)

        for i in range(len(log_freqs)):
            if i == 0:
                # First band: extend to halfway to next band
                log_width = (log_freqs[i + 1] - log_freqs[i])
            elif i == len(log_freqs) - 1:
                # Last band: extend to halfway from previous band
                log_width = (log_freqs[i] - log_freqs[i - 1])
            else:
                # Middle bands: halfway to neighbors on each side
                log_width = (log_freqs[i + 1] - log_freqs[i - 1]) / 2

            # Convert log width to linear bandwidth
            freq_low = 10 ** (log_freqs[i] - log_width / 2)
            freq_high = 10 ** (log_freqs[i] + log_width / 2)
            bandwidths.append(freq_high - freq_low)

        return bandwidths

    def _analyze_band(
            self,
            left_fft: np.ndarray,
            right_fft: np.ndarray,
            left_mag: np.ndarray,
            right_mag: np.ndarray,
            combined_mag: np.ndarray,
            floor_linear: float
    ) -> Tuple[float, float, float]:
        """Analyze a frequency band and return pan, ipd, ic."""
        band_left_mag = np.mean(left_mag)
        band_right_mag = np.mean(right_mag)
        band_combined_mag = np.mean(combined_mag)

        if band_combined_mag < floor_linear:
            return 0.0, 0.0, 0.0

        # Calculate IID (pan)
        total = band_left_mag + band_right_mag
        if total > 1e-10:
            pan_value = (band_left_mag - band_right_mag) / total
        else:
            pan_value = 0.0
        pan_value = np.clip(pan_value, -1.0, 1.0)

        # IPD - Phase difference (averaged over band)
        left_phase = np.angle(left_fft)
        right_phase = np.angle(right_fft)
        phase_diff = left_phase - right_phase
        phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

        # Weighted average by magnitude
        weights = np.abs(left_fft) + np.abs(right_fft)
        if np.sum(weights) > 1e-10:
            ipd_value = np.average(phase_diff, weights=weights)
        else:
            ipd_value = 0.0

        # IC - Inter-channel Coherence
        cross_power = np.mean(left_fft * np.conj(right_fft))
        left_power = np.mean(np.abs(left_fft) ** 2)
        right_power = np.mean(np.abs(right_fft) ** 2)

        if left_power > 1e-10 and right_power > 1e-10:
            ic_value = np.abs(cross_power) / np.sqrt(left_power * right_power)
            ic_value = np.clip(ic_value, 0.0, 1.0)
        else:
            ic_value = 0.0

        return pan_value, ipd_value, ic_value

    def _analyze_bin(
            self,
            left_fft: complex,
            right_fft: complex,
            left_mag: float,
            right_mag: float,
            combined_mag: float,
            floor_linear: float
    ) -> Tuple[float, float, float]:
        """Analyze a single frequency bin and return pan, ipd, ic."""
        if combined_mag < floor_linear:
            return 0.0, 0.0, 0.0

        # Calculate IID (pan)
        total = left_mag + right_mag
        if total > 1e-10:
            pan_value = (left_mag - right_mag) / total
        else:
            pan_value = 0.0
        pan_value = np.clip(pan_value, -1.0, 1.0)

        # IPD - Phase difference
        left_phase = np.angle(left_fft)
        right_phase = np.angle(right_fft)
        phase_diff = left_phase - right_phase
        ipd_value = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

        # IC - Inter-channel Coherence
        cross_power = left_fft * np.conj(right_fft)
        left_power = np.abs(left_fft) ** 2
        right_power = np.abs(right_fft) ** 2

        if left_power > 1e-10 and right_power > 1e-10:
            ic_value = np.abs(cross_power) / np.sqrt(left_power * right_power)
            ic_value = np.clip(ic_value, 0.0, 1.0)
        else:
            ic_value = 0.0

        return pan_value, ipd_value, ic_value


def pack_stereo_metadata(pan_values, ipd_values, ic_values, min_freq, max_freq, point, n_bands, compress=True):
    """
    Pack PS metadata with optional Brotli compression:
    - pan_values: list of floats [-1, 1]
    - ipd_values: list of floats [-pi, pi]
    - ic_values: list of bools
    - min_freq: int (minimum frequency)
    - max_freq: int (maximum frequency)
    - point: int
    - n_bands: int (number of bands)
    - compress: bool (default True) - apply Brotli compression
    Returns: bytes
    """
    n = len(pan_values)
    if not (len(ipd_values) == len(ic_values) == n):
        raise ValueError("All input lists must have same length")

    if n != n_bands:
        raise ValueError(f"n_bands ({n_bands}) must match length of input lists ({n})")

    packed_bytes = bytearray()

    # Pack 4 integers at the beginning (4 bytes each = 16 bytes total)
    packed_bytes.extend(struct.pack('<i', min_freq))
    packed_bytes.extend(struct.pack('<i', max_freq))
    packed_bytes.extend(struct.pack('<i', point))
    packed_bytes.extend(struct.pack('<i', n_bands))

    # Pack PAN and IPD as int8
    for pan, ipd in zip(pan_values, ipd_values):
        pan_byte = int(round(pan * 127))
        ipd_byte = int(round(ipd / math.pi * 127))
        pan_byte = max(-128, min(127, pan_byte))
        ipd_byte = max(-128, min(127, ipd_byte))
        packed_bytes.append(pan_byte & 0xFF)
        packed_bytes.append(ipd_byte & 0xFF)

    # Pack IC as bits (1 bit per band)
    ic_byte = 0
    bit_count = 0
    for ic in ic_values:
        ic_byte = (ic_byte << 1) | (1 if ic else 0)
        bit_count += 1
        if bit_count == 8:
            packed_bytes.append(ic_byte & 0xFF)
            ic_byte = 0
            bit_count = 0

    # Remaining bits
    if bit_count > 0:
        ic_byte = ic_byte << (8 - bit_count)
        packed_bytes.append(ic_byte & 0xFF)

    raw_data = bytes(packed_bytes)

    if compress:
        # Apply Brotli compression (quality 11 = maximum compression)
        compressed = brotli.compress(raw_data, quality=11)
        # Add a 1-byte flag to indicate compression
        return b'\x01' + compressed
    else:
        # Add a 1-byte flag to indicate no compression
        return b'\x00' + raw_data


def unpack_stereo_metadata(packed_bytes):
    """
    Unpack PS metadata with automatic decompression
    Returns: (pan_values, ipd_values, ic_values, min_freq, max_freq, point, n_bands)
    """
    # Check compression flag
    compression_flag = packed_bytes[0]
    data = packed_bytes[1:]

    if compression_flag == 0x01:
        # Decompress
        data = brotli.decompress(data)
    elif compression_flag != 0x00:
        raise ValueError(f"Unknown compression flag: {compression_flag}")

    # Unpack 4 integers from the beginning
    min_freq = struct.unpack('<i', data[0:4])[0]
    max_freq = struct.unpack('<i', data[4:8])[0]
    point = struct.unpack('<i', data[8:12])[0]
    n_bands = struct.unpack('<i', data[12:16])[0]

    pan_values = []
    ipd_values = []
    ic_values = []

    # PAN/IPD start after the header (16 bytes)
    header_size = 16
    for i in range(n_bands):
        offset = header_size + i * 2
        pan_byte = struct.unpack('b', data[offset:offset + 1])[0]
        ipd_byte = struct.unpack('b', data[offset + 1:offset + 2])[0]
        pan = pan_byte / 127.0
        ipd = ipd_byte / 127.0 * math.pi
        pan_values.append(pan)
        ipd_values.append(ipd)

    # IC bits start after PAN/IPD
    ic_start = header_size + n_bands * 2
    total_ic_bits = n_bands
    bits_read = 0

    for b in data[ic_start:]:
        for i in range(7, -1, -1):
            if bits_read >= total_ic_bits:
                break
            bit = (b >> i) & 1
            ic_values.append(bool(bit))
            bits_read += 1

    return pan_values, ipd_values, ic_values, min_freq, max_freq, point