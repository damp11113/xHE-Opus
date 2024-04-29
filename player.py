import pyaudio
from libxheopus import DualOpusDecoder, XopusReader

# Initialize PyAudio
p = pyaudio.PyAudio()

decoder = DualOpusDecoder()

streamoutput = p.open(format=pyaudio.paInt16, channels=2, rate=48000, output=True)

xopusdecoder = XopusReader(r"test.xopus")

print(xopusdecoder.readmetadata())

try:
    for data in xopusdecoder.decode(decoder, True):
        streamoutput.write(data)

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    # Clean up PyAudio streams and terminate PyAudio
    streamoutput.stop_stream()
    streamoutput.close()
    p.terminate()