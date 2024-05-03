import pyaudio
from libxheopus import DualOpusDecoder, XopusReader
import argparse
from tqdm import tqdm
import wave

parser = argparse.ArgumentParser(description='xHE-Opus Decoder/Player')

parser.add_argument("input", help='Input xopus file path')
parser.add_argument("-o", "--output", help='Output wav int16 file path')

args = parser.parse_args()

progress = tqdm()

# Initialize PyAudio
p = pyaudio.PyAudio()

decoder = DualOpusDecoder()

xopusdecoder = XopusReader(args.input)

metadata = xopusdecoder.readmetadata()

progress.total = metadata["footer"]["length"]
print("\nloudness:", metadata["footer"]["contentloudness"], "DBFS")

print(metadata["header"])

if not args.output:
    progress.desc = "playing..."
    streamoutput = p.open(format=pyaudio.paInt16, channels=2, rate=48000, output=True)
    try:
        for data in xopusdecoder.decode(decoder, True):
            streamoutput.write(data)

            progress.update(1)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up PyAudio streams and terminate PyAudio
        streamoutput.stop_stream()
        streamoutput.close()
        p.terminate()

    progress.desc = "played"
else:
    progress.desc = "converting..."
    outwav = wave.open(args.output, "w")
    # Set the parameters of the WAV file
    outwav.setnchannels(2)  # Stereo
    outwav.setsampwidth(2)  # 2 bytes (16 bits) per sample
    outwav.setframerate(48000)
    for data in xopusdecoder.decode(decoder, True):
        # Write the audio data to the file
        outwav.writeframes(data)
        progress.update(1)

    progress.desc = "converted"