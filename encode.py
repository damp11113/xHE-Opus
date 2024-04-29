from libxheopus import DualOpusEncoder, XopusWriter
import wave

encoder = DualOpusEncoder("restricted_lowdelay", version="hev2")
encoder.set_bitrates(12000)
encoder.set_bitrate_mode("CVBR")
desired_frame_size = encoder.set_frame_size(120)

wav_file = wave.open(r"test.wav", 'rb')

file = r"test.xopus"

xopus = XopusWriter(file, encoder)

# Read and process the WAV file in chunks
print("encoding...")
while True:
    frames = wav_file.readframes(desired_frame_size)

    if not frames:
        break  # Break the loop when all frames have been read

    xopus.write(frames)

xopus.close()
print("encoded")