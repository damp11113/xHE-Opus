import numpy as np
import pyaudio
import os
from libxheopus import DualOpusEncoder, xOpusDecoder

encoder = DualOpusEncoder(samplerate=48000, version="hev2")
encoder.set_stereo_mode(2)
encoder.set_bitrates(32000, balance_percent=75)
encoder.set_bitrate_mode("CVBR")
encoder.set_bandwidth("fullband")

encoder.set_compression(10)
desired_frame_size = encoder.set_frame_size(120)

decoder = xOpusDecoder(48000)

p = pyaudio.PyAudio()

device_name_input = "Line 5 (Virtual Audio Cable)"
device_index_input = 0
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['name'] == device_name_input:
        device_index_input = dev['index']
        break

device_name_output = "Speakers (2- USB AUDIO DEVICE)"
device_index_output = 0
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['name'] == device_name_output:
        device_index_output = dev['index']
        break

def callback(in_data, frame_count, time_info, status):
    pcm = np.frombuffer(in_data, dtype=np.int16)

    encoded_packets = encoder.encode(pcm)

    print(len(pcm), "-encoded->", len(encoded_packets))

    decoded_pcm = decoder.decode(encoded_packets)


    # Check if the decoded PCM is empty or not
    if len(decoded_pcm) > 0:
        pcm_to_write = np.frombuffer(decoded_pcm, dtype=np.int16)

        print(pcm_to_write)

        return (pcm_to_write.astype(np.int16).tobytes(), pyaudio.paContinue)

    else:
        print("Decoded PCM is empty")
        return (b"\x00", pyaudio.paContinue)




stream = p.open(format=pyaudio.paInt16, channels=2, rate=48000,
                input=True, input_device_index=device_index_input,
                output=True, output_device_index=device_index_output,
                stream_callback=callback, frames_per_buffer=desired_frame_size)

stream.start_stream()

print("Streaming audio. Press Ctrl+C to stop.")
try:
    while stream.is_active():
        pass
except KeyboardInterrupt:
    print("Stopping stream...")

# Stop and close stream
stream.stop_stream()
stream.close()
p.terminate()