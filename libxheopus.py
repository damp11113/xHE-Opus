import importlib
import math
import struct
import pyogg
import os
import numpy as np
from scipy.signal import butter, filtfilt

def float32_to_int16(data_float32):
    data_int16 = (data_float32 * 32767).astype(np.int16)
    return data_int16

def int16_to_float32(data_int16):
    data_float32 = data_int16.astype(np.float32) / 32767.0
    return data_float32

class DualOpusEncoder:
    def __init__(self, app="restricted_lowdelay", samplerate=48000, version="stable"):
        """
        ----------------------------- version--------------------------
        hev2: libopus 1.5.1 (fre:ac)
        exper: libopus 1.5.1
        stable: libopus 1.4
        old: libopus 1.3.1
        custom: custom opus path you can use "pyogg_win_libopus_custom_path" env to change opus version (windows only)
        ------------------------- App----------------------------------

        Set the encoding mode.

        This must be one of 'voip', 'audio', or 'restricted_lowdelay'.

        'voip': Gives best quality at a given bitrate for voice
        signals. It enhances the input signal by high-pass
        filtering and emphasizing formants and
        harmonics. Optionally it includes in-band forward error
        correction to protect against packet loss. Use this mode
        for typical VoIP applications. Because of the enhancement,
        even at high bitrates the output may sound different from
        the input.

        'audio': Gives best quality at a given bitrate for most
        non-voice signals like music. Use this mode for music and
        mixed (music/voice) content, broadcast, and applications
        requiring less than 15 ms of coding delay.

        'restricted_lowdelay': configures low-delay mode that
        disables the speech-optimized mode in exchange for
        slightly reduced delay. This mode can only be set on an
        newly initialized encoder because it changes the codec
        delay.
        """
        self.version = version
        self.samplerate = samplerate
        self.stereomode = 1 #0 = mono, 1 = Stereo LR, 2 = Stereo Mid/Side, 3 = Stereo Intensity
        self.automonogate = -50
        self.automono = False
        self.msmono = False
        self.overallbitrate = 0
        self.secbitrate = 0
        self.intensity = 1
        self.bitratemode = 1 # 0 = CBR, 1 = CVBR, 2 = VBR

        os.environ["pyogg_win_libopus_version"] = version
        importlib.reload(pyogg.opus)

        self.Lencoder = pyogg.OpusBufferedEncoder()
        self.Rencoder = pyogg.OpusBufferedEncoder()

        self.Lencoder.set_application(app)
        self.Rencoder.set_application(app)

        self.Lencoder.set_sampling_frequency(samplerate)
        self.Rencoder.set_sampling_frequency(samplerate)

        self.Lencoder.set_channels(1)
        self.Rencoder.set_channels(1)

        self.set_frame_size()
        self.set_compression()
        self.set_feature()
        self.set_bitrate_mode()
        self.set_bitrates()
        self.set_bandwidth()
        self.set_packet_loss()

    def set_compression(self, level=10):
        """complex 0-10 low-hires"""
        self.Lencoder.set_compresion_complex(level)
        self.Rencoder.set_compresion_complex(level)

    def set_bitrates(self, bitrates=64000, samebitrate=False, balance_percent=None):
        """
        input birate unit: bps

        balance_percent is working good with M/S stereo
        """
        if bitrates <= 5000:
            bitrates = 5000

        if balance_percent is None:
            if self.stereomode == 0:
                balance_percent = 100
            elif self.stereomode == 2:
                balance_percent = 75
            else:
                balance_percent = 50

        self.overallbitrate = bitrates

        if samebitrate:
            self.Lencoder.set_bitrates(int(bitrates))
            self.Rencoder.set_bitrates(int(bitrates))
        else:
            percentage_decimal = balance_percent / 100
            bitratech1 = round(bitrates * percentage_decimal)
            bitratech2 = bitrates - bitratech1

            if bitratech1 < 2500:
                bitratech1 = 2500

            if bitratech2 < 2500:
                self.msmono = True
                bitratech2 = 2500
            else:
                self.msmono = False

            self.secbitrate = bitratech1

            self.Lencoder.set_bitrates(int(bitratech1))
            self.Rencoder.set_bitrates(int(bitratech2))

    def set_bandwidth(self, bandwidth="fullband"):
        """
        narrowband:
        Narrowband typically refers to a limited range of frequencies suitable for voice communication.

        mediumband (unsupported in libopus 1.3+):
        Mediumband extends the frequency range compared to narrowband, providing better audio quality.

        wideband:
        Wideband offers an even broader frequency range, resulting in higher audio fidelity compared to narrowband and mediumband.

        superwideband:
        Superwideband extends the frequency range beyond wideband, further enhancing audio quality.

        fullband (default):
        Fullband provides the widest frequency range among the listed options, offering the highest audio quality.

        auto: opus is working auto not force
        """
        self.Lencoder.set_bandwidth(bandwidth)
        self.Rencoder.set_bandwidth(bandwidth)

    def set_stereo_mode(self, mode=1, automono=False, automonogate=-50, intensity=1, changebitratesbalance=False):
        """
        0 = mono (not recommend)
        1 = stereo LR
        2 = stereo Mid/Side
        3 = Intensity
        """
        if mode > 2:
            mode = 1

        self.stereomode = mode
        self.automono = automono
        self.automonogate = automonogate
        self.intensity = intensity

        if changebitratesbalance:
            self.set_bitrates(self.overallbitrate)

    def set_frame_size(self, size=60, nocheck=False):
        """ Set the desired frame duration (in milliseconds).
        Valid options are 2.5, 5, 10, 20, 40, or 60ms.
        Exclusive for HE opus v2 (freac opus) 80, 100 or 120ms.

        @return chunk size
        """
        if self.version != "hev2" and size > 60:
            raise ValueError("non hev2 can't use framesize > 60")

        self.Lencoder.set_frame_size(size, nocheck)
        self.Rencoder.set_frame_size(size, nocheck)

        return int((size / 1000) * self.samplerate)

    def set_packet_loss(self, loss=0):
        """input: % percent"""
        if loss > 100:
            raise ValueError("percent must <=100")

        self.Lencoder.set_packets_loss(loss)
        self.Rencoder.set_packets_loss(loss)

    def set_bitrate_mode(self, mode="CVBR"):
        """VBR, CVBR, CBR
        VBR in 1.5.x replace by CVBR
        """
        if mode.lower() == "cbr":
            self.bitratemode = 0
        elif mode.lower() == "cvbr":
            self.bitratemode = 1
        elif mode.lower() == "vbr":
            self.bitratemode = 2
        else:
            raise ValueError(f"No {mode} bitrate mode option")

        self.Lencoder.set_bitrate_mode(mode)
        self.Rencoder.set_bitrate_mode(mode)

    def set_feature(self, prediction=False, phaseinvert=False, DTX=False):
        self.Lencoder.CTL(pyogg.opus.OPUS_SET_PREDICTION_DISABLED_REQUEST, int(prediction))
        self.Lencoder.CTL(pyogg.opus.OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, int(phaseinvert))
        self.Lencoder.CTL(pyogg.opus.OPUS_SET_DTX_REQUEST, int(DTX))

        self.Rencoder.CTL(pyogg.opus.OPUS_SET_PREDICTION_DISABLED_REQUEST, int(prediction))
        self.Rencoder.CTL(pyogg.opus.OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, int(phaseinvert))
        self.Rencoder.CTL(pyogg.opus.OPUS_SET_DTX_REQUEST, int(DTX))

    def enable_voice_mode(self, enable=True, auto=False):
        self.Lencoder.enable_voice_enhance(enable, auto)
        self.Rencoder.enable_voice_enhance(enable, auto)

    def encode(self, pcmbytes, directpcm=False):
        """input: pcm bytes accept float32/int16 only
        x74 is mono
        x75 is stereo LR
        x76 is stereo mid/side

        xnl is no side audio
        """
        if directpcm:
            if pcmbytes.dtype == np.float32:
                pcm = (pcmbytes * 32767).astype(np.int16)
            elif pcmbytes.dtype == np.int16:
                pcm = pcmbytes.astype(np.int16)
            else:
                raise TypeError("accept only int16/float32")
        else:
            pcm = np.frombuffer(pcmbytes, dtype=np.int16)

        if self.stereomode == 0:
            # mono
            left_channel = pcm[::2]
            right_channel = pcm[1::2]
            mono = (left_channel + right_channel) / 2

            intmono = float32_to_int16(mono)

            midencoded_packet = self.Lencoder.buffered_encode(memoryview(bytearray(intmono)), flush=True)[0][0].tobytes()

            dual_encoded_packet = (midencoded_packet + b'\\x64\\x74')
        elif self.stereomode == 2:
            # stereo mid/side (Joint encoding)
            # convert to float32
            pcm = int16_to_float32(pcm)

            left_channel = pcm[::2]
            right_channel = pcm[1::2]

            mid = (left_channel + right_channel) / 2
            side = (left_channel - right_channel) / 2

            # convert back to int16
            mid = float32_to_int16(mid)
            intside = float32_to_int16(side)

            # check if side is no audio or loudness <= -50 DBFS
            try:
                loudnessside = 20 * math.log10(np.sqrt(np.mean(np.square(side))))
            except:
                loudnessside = 0

            if (loudnessside) <= self.automonogate and self.automono or self.msmono:
                sideencoded_packet = b"\\xnl"
                if self.bitratemode == 0: # CBR
                    self.Lencoder.set_bitrates(int(self.overallbitrate - 300))
            else:
                self.Lencoder.set_bitrates(int(self.secbitrate))
                sideencoded_packet = self.Rencoder.buffered_encode(memoryview(bytearray(intside)), flush=True)[0][0].tobytes()

            midencoded_packet = self.Lencoder.buffered_encode(memoryview(bytearray(mid)), flush=True)[0][0].tobytes()

            dual_encoded_packet = (midencoded_packet + b'\\x64\\x76' + sideencoded_packet)
        elif self.stereomode == 3:
            # stereo intensity (Joint encoding)
            left_channel = pcm[:, 0]
            right_channel = pcm[:, 1]

            IRChannel = left_channel + self.intensity * (right_channel - left_channel)

            Lencoded_packet = self.Rencoder.buffered_encode(memoryview(bytearray(left_channel)), flush=True)[0][0].tobytes()

            IRencoded_packet = self.Lencoder.buffered_encode(memoryview(bytearray(IRChannel)), flush=True)[0][0].tobytes()

            dual_encoded_packet = (Lencoded_packet + b'\\x64\\x77' + IRencoded_packet)
        else:
            # stereo LR
            left_channel = pcm[::2]
            right_channel = pcm[1::2]

            Lencoded_packet = self.Lencoder.buffered_encode(memoryview(bytearray(left_channel)), flush=True)[0][0].tobytes()
            Rencoded_packet = self.Rencoder.buffered_encode(memoryview(bytearray(right_channel)), flush=True)[0][0].tobytes()

            dual_encoded_packet = (Lencoded_packet + b'\\x64\\x75' + Rencoded_packet)

        return dual_encoded_packet

class PSOpusEncoder:
    def __init__(self, app="restricted_lowdelay", samplerate=48000, version="stable"):
        """
        This version is xHE-Opus v2 (Parametric Stereo)
        ----------------------------- version--------------------------
        hev2: libopus 1.5.1 (fre:ac)
        exper: libopus 1.5.1
        stable: libopus 1.4
        old: libopus 1.3.1
        custom: custom opus path you can use "pyogg_win_libopus_custom_path" env to change opus version (windows only)
        ------------------------- App----------------------------------

        Set the encoding mode.

        This must be one of 'voip', 'audio', or 'restricted_lowdelay'.

        'voip': Gives best quality at a given bitrate for voice
        signals. It enhances the input signal by high-pass
        filtering and emphasizing formants and
        harmonics. Optionally it includes in-band forward error
        correction to protect against packet loss. Use this mode
        for typical VoIP applications. Because of the enhancement,
        even at high bitrates the output may sound different from
        the input.

        'audio': Gives best quality at a given bitrate for most
        non-voice signals like music. Use this mode for music and
        mixed (music/voice) content, broadcast, and applications
        requiring less than 15 ms of coding delay.

        'restricted_lowdelay': configures low-delay mode that
        disables the speech-optimized mode in exchange for
        slightly reduced delay. This mode can only be set on an
        newly initialized encoder because it changes the codec
        delay.
        """
        self.version = version
        self.samplerate = samplerate

        os.environ["pyogg_win_libopus_version"] = version
        importlib.reload(pyogg.opus)

        self.encoder = pyogg.OpusBufferedEncoder()

        self.encoder.set_application(app)

        self.encoder.set_sampling_frequency(samplerate)

        self.encoder.set_channels(1)

        self.set_frame_size()
        self.set_compression()
        self.set_feature()
        self.set_bitrate_mode()
        self.set_bitrates()
        self.set_bandwidth()
        self.set_packet_loss()

    def set_compression(self, level=10):
        """complex 0-10 low-hires"""
        self.encoder.set_compresion_complex(level)

    def set_bitrates(self, bitrates=64000):
        """input birate unit: bps"""
        if bitrates <= 2500:
            bitrates = 2500

        self.encoder.set_bitrates(bitrates)

    def set_bandwidth(self, bandwidth="fullband"):
        """
        narrowband:
        Narrowband typically refers to a limited range of frequencies suitable for voice communication.

        mediumband (unsupported in libopus 1.3+):
        Mediumband extends the frequency range compared to narrowband, providing better audio quality.

        wideband:
        Wideband offers an even broader frequency range, resulting in higher audio fidelity compared to narrowband and mediumband.

        superwideband:
        Superwideband extends the frequency range beyond wideband, further enhancing audio quality.

        fullband (default):
        Fullband provides the widest frequency range among the listed options, offering the highest audio quality.

        auto: opus is working auto not force
        """
        self.encoder.set_bandwidth(bandwidth)

    def set_frame_size(self, size=60):
        """ Set the desired frame duration (in milliseconds).
        Valid options are 2.5, 5, 10, 20, 40, or 60ms.
        Exclusive for HE opus v2 (freac opus) 80, 100 or 120ms.

        @return chunk size
        """
        if self.version != "hev2" and size > 60:
            raise ValueError("non hev2 can't use framesize > 60")

        self.encoder.set_frame_size(size)

        return int((size / 1000) * self.samplerate)

    def set_packet_loss(self, loss=0):
        """input: % percent"""
        if loss > 100:
            raise ValueError("percent must <=100")

        self.encoder.set_packets_loss(loss)

    def set_bitrate_mode(self, mode="CVBR"):
        """VBR, CVBR, CBR
        VBR in 1.5.x replace by CVBR
        """

        self.encoder.set_bitrate_mode(mode)

    def set_feature(self, prediction=False, phaseinvert=False, DTX=False):
        self.encoder.CTL(pyogg.opus.OPUS_SET_PREDICTION_DISABLED_REQUEST, int(prediction))
        self.encoder.CTL(pyogg.opus.OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, int(phaseinvert))
        self.encoder.CTL(pyogg.opus.OPUS_SET_DTX_REQUEST, int(DTX))

    def enable_voice_mode(self, enable=True, auto=False):
        self.encoder.enable_voice_enhance(enable, auto)

    def __parameterization(self, stereo_signal):
        # Convert int16 to float32 for processing
        stereo_signal = stereo_signal.astype(np.float32) / 32768.0

        # Reshape stereo_signal into a 2D array with two channels
        stereo_signal = stereo_signal.reshape((-1, 2))

        # Calculate the magnitude spectrogram for each channel
        mag_left = np.abs(np.fft.fft(stereo_signal[:, 0]))
        mag_right = np.abs(np.fft.fft(stereo_signal[:, 1]))

        # Calculate the phase difference between the left and right channels
        phase_diff = np.angle(stereo_signal[:, 0]) - np.angle(stereo_signal[:, 1])

        # Compute other spatial features
        # Calculate stereo width
        stereo_width = np.mean(np.correlate(mag_left, mag_right, mode='full'))

        # Calculate phase coherence
        phase_coherence = np.mean(np.cos(phase_diff))

        # Calculate stereo panning
        stereo_panning_left = np.mean(mag_left / (mag_left + mag_right))
        stereo_panning_right = np.mean(mag_right / (mag_left + mag_right))

        pan = stereo_panning_right - stereo_panning_left

        # Return the derived parameters
        return (int(stereo_width), phase_coherence, pan)

    def encode(self, pcmbytes, directpcm=False):
        """input: pcm bytes accept float32/int16 only
        x74 is mono
        x75 is stereo LR
        x76 is stereo mid/side

        xnl is no side audio
        """
        if directpcm:
            if pcmbytes.dtype == np.float32:
                pcm = (pcmbytes * 32767).astype(np.int16)
            elif pcmbytes.dtype == np.int16:
                pcm = pcmbytes.astype(np.int16)
            else:
                raise TypeError("accept only int16/float32")
        else:
            pcm = np.frombuffer(pcmbytes, dtype=np.int16)

        pcmreshaped = pcm.reshape(-1, 2)

        mono_data = np.mean(pcmreshaped * 0.5, axis=1, dtype=np.int16)

        stereodata = self.__parameterization(pcmreshaped)
        packedstereodata = struct.pack('iff', *stereodata)

        encoded_packet = self.encoder.buffered_encode(memoryview(bytearray(mono_data)), flush=True)[0][0].tobytes()

        encoded_packet = (encoded_packet + b'\\x21\\x75' + packedstereodata)

        return encoded_packet

class xOpusDecoder:
    def __init__(self, sample_rate=48000):
        self.Ldecoder = pyogg.OpusDecoder()
        self.Rdecoder = pyogg.OpusDecoder()

        self.Ldecoder.set_channels(1)
        self.Rdecoder.set_channels(1)

        self.Ldecoder.set_sampling_frequency(sample_rate)
        self.Rdecoder.set_sampling_frequency(sample_rate)

        self.__prev_pan = 0.0
        self.__prev_max_amplitude = 0.0

    def __smooth(self, value, prev_value, alpha=0.1):
        return alpha * value + (1 - alpha) * prev_value

    def __expand_and_pan(self, input_signal, pan_value, expansion_factor, gain):
        """
        Apply stereo expansion and panning to an input audio signal.

        Parameters:
        - input_signal: Input audio signal (numpy array of int16).
        - expansion_factor: Factor to expand the stereo width (0 to 1).
        - pan_value: Pan value (-1 to 1, where -1 is full left, 1 is full right).
        - gain: Gain factor to adjust the volume.

        Returns:
        - output_signal: Processed audio signal (stereo, numpy array of int16).
        """

        # Convert int16 to float32 for processing
        input_signal_float = input_signal.astype(np.float32) / 32768.0

        # Separate the channels
        left_channel = input_signal_float[:, 0]
        right_channel = input_signal_float[:, 1]

        # Apply panning
        pan_left = (1 - pan_value) / 2
        pan_right = (1 + pan_value) / 2
        left_channel *= pan_left
        right_channel *= pan_right

        # Apply stereo expansion
        center = (left_channel + right_channel) / 2
        left_channel = center + (left_channel - center) * expansion_factor
        right_channel = center + (right_channel - center) * expansion_factor

        # Apply gain
        left_channel *= gain
        right_channel *= gain

        # Ensure no clipping by normalizing if necessary
        max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        if max_val > 1.0:
            left_channel /= max_val
            right_channel /= max_val

        # Merge the channels
        output_signal = np.stack((left_channel, right_channel), axis=-1)

        return (output_signal * 32767).astype(np.int16)

    def __mix_stereo_signals(self, signal1, signal2, volume1=1.0, volume2=1.0):
        # Ensure both signals have the same length
        length = max(len(signal1), len(signal2))
        signal1 = np.pad(signal1, ((0, length - len(signal1)), (0, 0)), mode='constant')
        signal2 = np.pad(signal2, ((0, length - len(signal2)), (0, 0)), mode='constant')

        # Convert signals to float
        signal1 = signal1.astype(np.float32)
        signal2 = signal2.astype(np.float32)

        # Adjust volume
        signal1 *= volume1
        signal2 *= volume2

        # Mix the signals
        mixed_signal = signal1 + signal2

        # Normalize the mixed signal to prevent clipping
        max_amplitude = np.max(np.abs(mixed_signal))
        if max_amplitude > 32767:
            mixed_signal = (mixed_signal / max_amplitude) * 32767

        return mixed_signal.astype(np.int16)

    def __apply_smoothing_window(self, audio_data, window_size):
        """
        Apply a smoothing window to the beginning and end of the audio data.

        Parameters:
        - audio_data: 2D numpy array with shape (num_samples, 2)
        - window_size: Size of the smoothing window in samples

        Returns:
        - smoothed_audio_data: 2D numpy array with the smoothing window applied
        """
        window = np.hanning(window_size * 2)
        fade_in = window[:window_size]
        fade_out = window[-window_size:]

        audio_data[:window_size, :] *= fade_in[:, np.newaxis]
        audio_data[-window_size:, :] *= fade_out[:, np.newaxis]

        return audio_data

    def __stereo_widening_effect(self, data, delay_samples=10, gain=0.8, window_size=100):
        audio_data = data.reshape(-1, 2)

        # Convert int16 to float32 for processing
        audio_data = audio_data.astype(np.float32)

        # Apply delay to the right channel
        right_channel = np.roll(audio_data[:, 1], delay_samples)

        # Apply gain to both channels
        audio_data[:, 0] *= gain
        right_channel *= gain

        # Combine channels back into stereo
        widened_audio_data = np.stack((audio_data[:, 0], right_channel), axis=1)

        # Apply smoothing window to reduce clicks
        widened_audio_data = self.__apply_smoothing_window(widened_audio_data, window_size)

        # Clip to avoid overflow
        widened_audio_data = np.clip(widened_audio_data, -32768, 32767)

        # Convert float32 back to int16
        widened_audio_data = widened_audio_data.astype(np.int16)

        return widened_audio_data

    def __apply_phase_coherence_to_stereo(self, signal, phase_coherence):
        # Convert phase coherence to phase shift in radians
        phase_shift = np.arccos(phase_coherence)
        # Apply phase shift to both channels
        return self.__apply_phase_shift(signal, phase_shift)

    # Function to apply phase shift to one channel
    def __apply_phase_shift(self, signal, phase_shift):
        # Convert to complex
        signal_complex = signal.astype(np.complex64)
        # Apply phase shift
        shifted_signal = signal_complex * np.exp(1j * phase_shift)
        return shifted_signal.astype(np.int16)

    def __synthstereo(self, mono_signal, stereodata):
        pan = stereodata[2]

        # Smooth the pan value
        pan = self.__smooth(pan, self.__prev_pan, alpha=0.25)
        self.__prev_pan = pan

        stereo_exp = stereodata[0] / 10000

        try:
            delayed = self.__stereo_widening_effect(mono_signal, int(stereo_exp), 1, int(stereo_exp) * 2)
        except:
            delayed = mono_signal

        l1 = self.__expand_and_pan(mono_signal, pan, 1, 2)

        stereo_signal_shifted = self.__apply_phase_coherence_to_stereo(delayed, stereodata[1])

        return self.__mix_stereo_signals(l1, stereo_signal_shifted, volume1=1, volume2=0.5).astype(np.int16)

    def decode(self, dualopusbytes: bytes, outputformat=np.int16):
        # mode check
        if b"\\x64\\x74" in dualopusbytes:
            mode = 0
            xopusbytespilted = dualopusbytes.split(b'\\x64\\x74')
        elif b"\\x64\\x76" in dualopusbytes:
            mode = 2
            xopusbytespilted = dualopusbytes.split(b'\\x64\\x76')
        elif b"\\x64\\x75" in dualopusbytes:
            mode = 1
            xopusbytespilted = dualopusbytes.split(b'\\x64\\x75')
        elif b"\\x64\\x77" in dualopusbytes:
            mode = 4
            xopusbytespilted = dualopusbytes.split(b'\\x64\\x77')
        elif b"\\x21\\x75" in dualopusbytes:
            mode = 3 # v2
            xopusbytespilted = dualopusbytes.split(b'\\x21\\x75')
        else:
            raise TypeError("this is not xopus bytes")

        if mode == 0: # mono
            Mencoded_packet = xopusbytespilted[0]
            decoded_left_channel_pcm = self.Ldecoder.decode(memoryview(bytearray(Mencoded_packet)))
            Mpcm = np.frombuffer(decoded_left_channel_pcm, dtype=np.int16)

            stereo_signal = np.column_stack((Mpcm, Mpcm))
        elif mode == 2:
            # stereo mid/side (Joint encoding)
            Mencoded_packet = xopusbytespilted[0]
            Sencoded_packet = xopusbytespilted[1]

            decoded_mid_channel_pcm = self.Ldecoder.decode(memoryview(bytearray(Mencoded_packet)))
            Mpcm = np.frombuffer(decoded_mid_channel_pcm, dtype=np.int16)

            if Sencoded_packet != b"\\xnl":
                decoded_side_channel_pcm = self.Rdecoder.decode(memoryview(bytearray(Sencoded_packet)))
                Spcm = np.frombuffer(decoded_side_channel_pcm, dtype=np.int16)

                Mpcm = int16_to_float32(Mpcm)
                Spcm = int16_to_float32(Spcm)

                L = Mpcm + Spcm
                R = Mpcm - Spcm

                stereo_signal = np.column_stack((L, R))

                #max_amplitude = np.max(np.abs(stereo_signal))

                #if max_amplitude > 1.0:
                #    stereo_signal /= max_amplitude

                stereo_signal = np.clip(stereo_signal, -1, 1)

                stereo_signal = float32_to_int16(stereo_signal)
            else:
                stereo_signal = np.column_stack((Mpcm, Mpcm))
        elif mode == 4:
            # stereo intensity
            Lencoded_packet = xopusbytespilted[0]
            IRencoded_packet = xopusbytespilted[1]

            decoded_left_channel_pcm = self.Ldecoder.decode(memoryview(bytearray(Lencoded_packet)))
            decoded_intensity_right_channel_pcm = self.Rdecoder.decode(memoryview(bytearray(IRencoded_packet)))

            Lpcm = np.frombuffer(decoded_left_channel_pcm, dtype=np.int16)
            IRpcm = np.frombuffer(decoded_intensity_right_channel_pcm, dtype=np.int16)

            recovered_right = Lpcm + (IRpcm - Lpcm) / 1

            stereo_signal = np.column_stack((Lpcm, recovered_right))
        elif mode == 3:
            # Parametric Stereo
            Mencoded_packet = xopusbytespilted[0]
            stereodatapacked = xopusbytespilted[1]

            stereodata = struct.unpack('iff', stereodatapacked)

            mono_channel_pcm = self.Ldecoder.decode(memoryview(bytearray(Mencoded_packet)))
            Mpcm = np.frombuffer(mono_channel_pcm, dtype=np.int16)

            stereo_audio = np.stack((Mpcm, Mpcm)).T.reshape(-1, 2)

            stereo_signal = self.__synthstereo(stereo_audio, stereodata)
        else:
            # stereo LR
            Lencoded_packet = xopusbytespilted[0]
            Rencoded_packet = xopusbytespilted[1]

            decoded_left_channel_pcm = self.Ldecoder.decode(memoryview(bytearray(Lencoded_packet)))
            decoded_right_channel_pcm = self.Rdecoder.decode(memoryview(bytearray(Rencoded_packet)))

            Lpcm = np.frombuffer(decoded_left_channel_pcm, dtype=np.int16)
            Rpcm = np.frombuffer(decoded_right_channel_pcm, dtype=np.int16)

            stereo_signal = np.column_stack((Lpcm, Rpcm))

        return stereo_signal.astype(outputformat).tobytes()

class HeaderContainer:
    def __init__(self, capture_pattern, version, metadata):
        self.capture_pattern = capture_pattern
        self.version = version
        self.metadata = metadata

    def serialize(self):
        header = struct.pack('<4sB', self.capture_pattern, self.version)
        metadata_bytes = self.serialize_metadata()
        return header + metadata_bytes

    def serialize_metadata(self):
        metadata_bytes = b''
        for key, value in self.metadata.items():
            key_bytes = key.encode('utf-8')
            value_bytes = value.encode('utf-8') if isinstance(value, str) else str(value).encode('utf-8')
            metadata_bytes += struct.pack(f'<I{len(key_bytes)}sI{len(value_bytes)}s', len(key_bytes), key_bytes, len(value_bytes), value_bytes)
        return metadata_bytes

    @classmethod
    def deserialize(cls, data):
        capture_pattern, version = struct.unpack_from('<4sB', data)
        metadata_start = struct.calcsize('<4sB')
        metadata = cls.deserialize_metadata(data[metadata_start:])
        return cls(capture_pattern, version, metadata)

    @staticmethod
    def deserialize_metadata(metadata_bytes):
        metadata = {}
        while metadata_bytes:
            key_length = struct.unpack('<I', metadata_bytes[:4])[0]
            key = struct.unpack(f'<{key_length}s', metadata_bytes[4:4+key_length])[0].decode('utf-8')
            metadata_bytes = metadata_bytes[4+key_length:]
            value_length = struct.unpack('<I', metadata_bytes[:4])[0]
            value = struct.unpack(f'<{value_length}s', metadata_bytes[4:4+value_length])[0].decode('utf-8')
            metadata_bytes = metadata_bytes[4+value_length:]
            metadata[key] = value
        return metadata

class FooterContainer:
    def __init__(self, loudness_avg, length):
        self.loudness_avg = loudness_avg
        self.length = length

    def serialize(self):
        metadata_bytes = self.serialize_metadata()
        return metadata_bytes

    def serialize_metadata(self):
        metadata_bytes = b''
        metadata_bytes += struct.pack('<f', self.loudness_avg)
        metadata_bytes += struct.pack('<I', self.length)
        return metadata_bytes

    @classmethod
    def deserialize(cls, data):
        loudness_avg, length = cls.deserialize_metadata(data)
        return cls(loudness_avg, length)

    @staticmethod
    def deserialize_metadata(metadata_bytes):
        loudness_avg = struct.unpack('<f', metadata_bytes[:4])[0]
        length = struct.unpack('<I', metadata_bytes[4:8])[0]
        return loudness_avg, length

class XopusWriter:
    def __init__(self, file, encoder: DualOpusEncoder, metadata=None):
        self.file = file
        self.encoder = encoder

        if metadata is None:
            metadata = {}

        systemmetadata = {
            "format": "Xopus",
            "audio": {
                "encoder": "libxheopus",
                "format": "xHE-Opus",
                "format/info": "Extended High Efficiency Opus Audio Codec"
            }
        }

        open(file, 'wb').write(b"") # clear
        self.xopusfile = open(file, 'ab')
        self.xopusfile.write(HeaderContainer(b'OpuS', 1, metadata | systemmetadata).serialize() + b"\\xa")

        self.loudnessperframe = []
        self.length = 0

    def write(self, pcmbytes):
        pcm = np.frombuffer(pcmbytes, dtype=np.int16)
        # Convert int16 audio data to floating-point values in range [-1, 1]
        normalized_audio = pcm / 32767.0

        # Calculate RMS value
        rms = np.sqrt(np.mean(np.square(normalized_audio)))

        # Calculate dBFS
        try:
            dbfs = 20 * math.log10(rms)
        except:
            dbfs = 0
        self.loudnessperframe.append(dbfs)

        encoded = self.encoder.encode(pcm, directpcm=True)
        self.xopusfile.write(encoded + b"\\xa")
        self.length += 1

    def close(self):
        loudnessavgs = sum(self.loudnessperframe) / len(self.loudnessperframe)

        self.xopusfile.write(b"\\xeof\\xeof")
        self.xopusfile.write(FooterContainer(loudnessavgs, self.length).serialize())
        self.loudnessperframe = []
        self.length = 0

class XopusReader:
    def __init__(self, file):
        self.file = open(file, 'rb')
        self.xopusline = self.file.read().split(b"\\xa")
        self.lastframe = b""

    def readmetadata(self):
        header = HeaderContainer.deserialize(self.xopusline[0])

        if self.xopusline[-1].startswith(b"\\xeof\\xeof"):
            footer = FooterContainer.deserialize(self.xopusline[-1].split(b"\\xeof\\xeof")[1])
        else:
            raise EOFError("can't find EOF")

        data = {
            "header": dict(header.metadata),
            "footer": {
                "contentloudness": footer.loudness_avg,
                "length": footer.length
            }
        }
        return data

    def decode(self, decoder, play=False, start=0):
        if play:
            for data in self.xopusline[start + 1:]:
                if data.startswith(b"\\xeof\\xeof"):
                    break
                else:
                    try:
                        decodeddata = decoder.decode(data)
                        self.lastframe = decodeddata
                        yield decodeddata
                    except Exception as e:
                        #print(e)
                        yield self.lastframe
        else:
            decodedlist = []
            for data in self.xopusline[1:]:
                if data.startswith(b"\\xeof\\xeof"):
                    break
                else:
                    try:
                        decodeddata = decoder.decode(data)
                        self.lastframe = decodeddata
                        decodedlist.append(self.lastframe)
                    except:
                        decodedlist.append(self.lastframe)
            return decodedlist

    def close(self):
        self.xopusline = []
        self.file.close()