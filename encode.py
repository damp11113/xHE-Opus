from libxheopus import DualOpusEncoder, CustomFileContainer
import wave

encoder = DualOpusEncoder("restricted_lowdelay", version="hev2")
encoder.set_bitrates(12000)
encoder.set_bitrate_mode("CVBR")
desired_frame_size = encoder.set_frame_size(120)

wav_file = wave.open(r"C:\Users\sansw\Desktop\The Weeknd - Blinding Lights (HD+).wav", 'rb')

metadata = {"Format": "xHE-Opus", "loudness": 0}  # Replace with your metadata
container = CustomFileContainer(b'OpuS', 1, metadata)

file = r"test.xopus"

open(file, 'wb').write(b"") # clear
xopusfile = open(file, 'ab')
xopusfile.write(container.serialize() + b"\\xa")

# Read and process the WAV file in chunks
print("encoding...")
while True:
    frames = wav_file.readframes(desired_frame_size)

    encoded = encoder.encode(frames)

    if not frames:
        break  # Break the loop when all frames have been read

    xopusfile.write(encoded + b"\\xa")
    # Process the frames here, for example, print the number of bytes read


print("encoded")