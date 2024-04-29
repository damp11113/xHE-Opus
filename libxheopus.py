import importlib
import math
import struct

import pyogg
import os
import numpy as np

class DualOpusEncoder:
    def __init__(self, app="audio", samplerate=48000, version="stable"):
        """
        ----------------------------- version--------------------------
        hev2: libopus 1.5.1 (fre:ac)
        he: libopus 1.5.2 (moded)
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

    def set_bitrates(self, bitrates=64000, samebitrate=False):
        """input birate unit: bps"""
        if bitrates <= 5000:
            bitrates = 5000

        if samebitrate:
            bitperchannel = bitrates
        else:
            bitperchannel = bitrates / 2

        self.Lencoder.set_bitrates(int(bitperchannel))
        self.Rencoder.set_bitrates(int(bitperchannel))

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

    def set_frame_size(self, size=60):
        """ Set the desired frame duration (in milliseconds).
        Valid options are 2.5, 5, 10, 20, 40, or 60ms.
        Exclusive for HE opus v2 (freac opus) 80, 100 or 120ms.

        @return chunk size
        """

        if self.version != "hev2" and size > 60:
            raise ValueError("non hev2 can't use framesize > 60")

        self.Lencoder.set_frame_size(size)
        self.Rencoder.set_frame_size(size)

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

        self.Lencoder.set_bitrate_mode(mode)
        self.Rencoder.set_bitrate_mode(mode)

    def set_feature(self, prediction=False, phaseinvert=False, DTX=False):
        self.Lencoder.CTL(pyogg.opus.OPUS_SET_PREDICTION_DISABLED_REQUEST, int(prediction))
        self.Lencoder.CTL(pyogg.opus.OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, int(phaseinvert))
        self.Lencoder.CTL(pyogg.opus.OPUS_SET_DTX_REQUEST, int(DTX))

        self.Rencoder.CTL(pyogg.opus.OPUS_SET_PREDICTION_DISABLED_REQUEST, int(prediction))
        self.Rencoder.CTL(pyogg.opus.OPUS_SET_PHASE_INVERSION_DISABLED_REQUEST, int(phaseinvert))
        self.Rencoder.CTL(pyogg.opus.OPUS_SET_DTX_REQUEST, int(DTX))

    def encode(self, pcmbytes, directpcm=False):
        """input: pcm bytes accept float32/int16 only"""
        if directpcm:
            if pcmbytes.dtype == np.float32:
                pcm = (pcmbytes * 32767).astype(np.int16)
            elif pcmbytes.dtype == np.int16:
                pcm = pcmbytes.astype(np.int16)
            else:
                raise TypeError("accept only int16/float32")
        else:
            pcm = np.frombuffer(pcmbytes, dtype=np.int16)

        left_channel = pcm[::2]
        right_channel = pcm[1::2]

        Lencoded_packet = self.Lencoder.buffered_encode(memoryview(bytearray(left_channel)), flush=True)[0][0].tobytes()
        Rencoded_packet = self.Rencoder.buffered_encode(memoryview(bytearray(right_channel)), flush=True)[0][
            0].tobytes()

        dual_encoded_packet = (Lencoded_packet + b'\\x64\\x75' + Rencoded_packet)

        return dual_encoded_packet


class DualOpusDecoder:
    def __init__(self, sample_rate=48000):
        self.Ldecoder = pyogg.OpusDecoder()
        self.Rdecoder = pyogg.OpusDecoder()

        self.Ldecoder.set_channels(1)
        self.Rdecoder.set_channels(1)

        self.Ldecoder.set_sampling_frequency(sample_rate)
        self.Rdecoder.set_sampling_frequency(sample_rate)

    def decode(self, dualopusbytes: bytes, outputformat=np.int16):
        try:
            dualopusbytespilted = dualopusbytes.split(b'\\x64\\x75')
            Lencoded_packet = dualopusbytespilted[0]
            Rencoded_packet = dualopusbytespilted[1]
        except:
            raise TypeError("this is not dual opus")

        decoded_left_channel_pcm = self.Ldecoder.decode(memoryview(bytearray(Lencoded_packet)))
        decoded_right_channel_pcm = self.Rdecoder.decode(memoryview(bytearray(Rencoded_packet)))

        Lpcm = np.frombuffer(decoded_left_channel_pcm, dtype=outputformat)
        Rpcm = np.frombuffer(decoded_right_channel_pcm, dtype=outputformat)

        stereo_signal = np.empty((len(Lpcm), 2), dtype=Lpcm.dtype)
        stereo_signal[:, 0] = Lpcm
        stereo_signal[:, 1] = Rpcm

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
    def __init__(self, file, encoder: DualOpusEncoder, metadata={}):
        self.file = file
        self.encoder = encoder

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
        dbfs = 20 * math.log10(rms)
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
        file = open(file, 'rb')
        self.xopusline = file.read().split(b"\\xa")

    def readmetadata(self):
        header = HeaderContainer.deserialize(self.xopusline[0])

        if self.xopusline[-1].startswith(b"\\xeof\\xeof"):
            footer = FooterContainer.deserialize(self.xopusline[-1].split(b"\\xeof\\xeof")[1])
        else:
            raise EOFError("can't find EOF")

        data = {
            "header": header.metadata,
            "footer": {
                "contentloudness": footer.loudness_avg,
                "length": footer.length
            }
        }
        return data

    def decode(self, decoder, play=False):
        if play:
            for data in self.xopusline[1:]:
                if data.startswith(b"\\xeof\\xeof"):
                    break
                else:
                     yield decoder.decode(data)
        else:
            decodedlist = []
            for data in self.xopusline[1:]:
                if data.startswith(b"\\xeof\\xeof"):
                    break
                else:
                    decodedlist.append(decoder.decode(data))
            return decodedlist