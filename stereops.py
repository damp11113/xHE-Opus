import math
import struct
from typing import Tuple, List

import numpy as np


def stereo_frequency_analysis(
        audio_data: np.ndarray,
        sample_rate: int,
        min_freq: float,
        max_freq: float,
        freq_points: int,
        floor_db: float = -75.0,
        use_grouping: bool = False
) -> List[Tuple[float, float, float, float]]:
    """Analyze stereo audio and extract IID (pan), IPD, and IC per frequency band.

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
    audio_float = audio_data.astype(np.float32) / 32768.0

    # Separate channels
    left = audio_float[:, 0]
    right = audio_float[:, 1]

    # Calculate FFT for both channels
    n_fft = len(left)
    left_fft = np.fft.rfft(left)
    right_fft = np.fft.rfft(right)

    # Get frequency bins
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)

    # Calculate magnitude for each channel
    left_mag = np.abs(left_fft)
    right_mag = np.abs(right_fft)

    # Calculate combined magnitude for floor detection
    combined_mag = (left_mag + right_mag) / 2

    # Convert floor from dB to linear scale
    floor_linear = 10 ** (floor_db / 20.0)

    # Generate target frequencies to sample
    target_freqs = np.linspace(min_freq, max_freq, freq_points)

    # Calculate bandwidth for grouping
    if use_grouping and freq_points > 1:
        bandwidth = (max_freq - min_freq) / freq_points
    else:
        bandwidth = 0

    # Find nearest frequency bin indices
    results = []
    for target_freq in target_freqs:
        if use_grouping and bandwidth > 0:
            freq_low = target_freq - bandwidth / 2
            freq_high = target_freq + bandwidth / 2
            mask = (freqs >= freq_low) & (freqs <= freq_high)

            if np.any(mask):
                # IID (Pan) - Intensity difference
                band_left_mag = np.mean(left_mag[mask])
                band_right_mag = np.mean(right_mag[mask])
                band_combined_mag = np.mean(combined_mag[mask])

                if band_combined_mag < floor_linear:
                    pan_value = 0.0
                    ipd_value = 0.0
                    ic_value = 0.0
                else:
                    # Calculate IID (pan)
                    total = band_left_mag + band_right_mag
                    if total > 1e-10:
                        pan_value = (band_left_mag - band_right_mag) / total
                    else:
                        pan_value = 0.0
                    pan_value = np.clip(pan_value, -1.0, 1.0)

                    # IPD - Phase difference (averaged over band)
                    band_left_fft = left_fft[mask]
                    band_right_fft = right_fft[mask]

                    # Calculate phase for each FFT bin in band
                    left_phase = np.angle(band_left_fft)
                    right_phase = np.angle(band_right_fft)

                    # Phase difference
                    phase_diff = left_phase - right_phase

                    # Wrap to [-π, π]
                    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

                    # Weighted average by magnitude
                    weights = np.abs(band_left_fft) + np.abs(band_right_fft)
                    if np.sum(weights) > 1e-10:
                        ipd_value = np.average(phase_diff, weights=weights)
                    else:
                        ipd_value = 0.0

                    # IC - Inter-channel Coherence
                    # Normalized cross-correlation in frequency domain
                    cross_power = np.mean(band_left_fft * np.conj(band_right_fft))
                    left_power = np.mean(np.abs(band_left_fft) ** 2)
                    right_power = np.mean(np.abs(band_right_fft) ** 2)

                    if left_power > 1e-10 and right_power > 1e-10:
                        ic_value = np.abs(cross_power) / np.sqrt(left_power * right_power)
                        ic_value = np.clip(ic_value, 0.0, 1.0)
                    else:
                        ic_value = 0.0
            else:
                pan_value = 0.0
                ipd_value = 0.0
                ic_value = 0.0
        else:
            idx = np.argmin(np.abs(freqs - target_freq))

            if combined_mag[idx] < floor_linear:
                pan_value = 0.0
                ipd_value = 0.0
                ic_value = 0.0
            else:
                # Calculate IID (pan)
                total = left_mag[idx] + right_mag[idx]
                if total > 1e-10:
                    pan_value = (left_mag[idx] - right_mag[idx]) / total
                else:
                    pan_value = 0.0
                pan_value = np.clip(pan_value, -1.0, 1.0)

                # IPD - Phase difference
                left_phase = np.angle(left_fft[idx])
                right_phase = np.angle(right_fft[idx])
                phase_diff = left_phase - right_phase

                # Wrap to [-π, π]
                ipd_value = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

                # IC - Inter-channel Coherence
                cross_power = left_fft[idx] * np.conj(right_fft[idx])
                left_power = np.abs(left_fft[idx]) ** 2
                right_power = np.abs(right_fft[idx]) ** 2

                if left_power > 1e-10 and right_power > 1e-10:
                    ic_value = np.abs(cross_power) / np.sqrt(left_power * right_power)
                    ic_value = np.clip(ic_value, 0.0, 1.0)
                else:
                    ic_value = 0.0

        results.append((float(target_freq), float(-pan_value), float(ipd_value), float(ic_value)))

    return results


def apply_stereo(
        mono_audio: np.ndarray,
        sample_rate: int,
        pan_values: list,
        ipd_values: list,
        ic_values: list,  # list of bools
        min_freq: float,
        max_freq: float,
        freq_points: int,
        use_grouping: bool = False
) -> np.ndarray:
    """Apply frequency-dependent IID (pan), IPD (phase), and IC (coherence) to mono audio."""

    # Normalize if int16
    if mono_audio.dtype == np.int16:
        mono_float = mono_audio.astype(np.float32) / 32768.0
        return_int16 = True
    else:
        mono_float = mono_audio.astype(np.float32)
        return_int16 = False

    # FFT
    mono_fft = np.fft.rfft(mono_float)
    freqs = np.fft.rfftfreq(len(mono_float), 1 / sample_rate)

    target_freqs = np.linspace(min_freq, max_freq, freq_points)

    if use_grouping and freq_points > 1:
        bandwidth = (max_freq - min_freq) / freq_points
    else:
        bandwidth = 0

    left_gain = np.ones(len(freqs), dtype=np.float32)
    right_gain = np.ones(len(freqs), dtype=np.float32)
    right_phase_shift = np.ones(len(freqs), dtype=np.complex64)

    for i, (freq, pan, ipd, ic) in enumerate(zip(target_freqs, pan_values, ipd_values, ic_values)):
        # Constant power panning
        angle = (-pan + 1.0) * np.pi / 4.0
        left_g = np.cos(angle)
        right_g = np.sin(angle)

        # If IC is False, blend toward mono
        if not ic:
            # Reduce stereo difference
            left_g = right_g = (left_g + right_g) / 2.0
            ipd = 0.0  # remove phase difference

        if use_grouping and bandwidth > 0:
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


def pack_stereo_metadata(pan_values, ipd_values, ic_values, min_freq, max_freq, point, n_bands):
    """
    Pack PS metadata:
    - pan_values: list of floats [-1, 1]
    - ipd_values: list of floats [-pi, pi]
    - ic_values: list of bools
    - min_freq: int (minimum frequency)
    - max_freq: int (maximum frequency)
    - point: int
    - n_bands: int (number of bands)
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

    return bytes(packed_bytes)


def unpack_stereo_metadata(packed_bytes):
    """
    Unpack PS metadata
    Returns: (pan_values, ipd_values, ic_values, min_freq, max_freq, point, n_bands)
    """
    # Unpack 4 integers from the beginning
    min_freq = struct.unpack('<i', packed_bytes[0:4])[0]
    max_freq = struct.unpack('<i', packed_bytes[4:8])[0]
    point = struct.unpack('<i', packed_bytes[8:12])[0]
    n_bands = struct.unpack('<i', packed_bytes[12:16])[0]

    pan_values = []
    ipd_values = []
    ic_values = []

    # PAN/IPD start after the header (16 bytes)
    header_size = 16
    for i in range(n_bands):
        offset = header_size + i * 2
        pan_byte = struct.unpack('b', packed_bytes[offset:offset + 1])[0]
        ipd_byte = struct.unpack('b', packed_bytes[offset + 1:offset + 2])[0]
        pan = pan_byte / 127.0
        ipd = ipd_byte / 127.0 * math.pi
        pan_values.append(pan)
        ipd_values.append(ipd)

    # IC bits start after PAN/IPD
    ic_start = header_size + n_bands * 2
    total_ic_bits = n_bands
    bits_read = 0

    for b in packed_bytes[ic_start:]:
        for i in range(7, -1, -1):
            if bits_read >= total_ic_bits:
                break
            bit = (b >> i) & 1
            ic_values.append(bool(bit))
            bits_read += 1

    return pan_values, ipd_values, ic_values, min_freq, max_freq, point
