import pyaudio
from libxheopus import DualOpusDecoder, CustomFileContainer

# Initialize PyAudio
p = pyaudio.PyAudio()

decoder = DualOpusDecoder()

streamoutput = p.open(format=pyaudio.paInt16, channels=2, rate=48000, output=True)

file = open(r"test.xopus", 'rb')

line = file.read().split(b"\\xa")

deserialized_container = CustomFileContainer.deserialize(line[0])
print(deserialized_container.metadata)

try:
    for data in line[1:]:
        if data:
            streamoutput.write(decoder.decode(data))


except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    # Clean up PyAudio streams and terminate PyAudio
    streamoutput.stop_stream()
    streamoutput.close()
    p.terminate()