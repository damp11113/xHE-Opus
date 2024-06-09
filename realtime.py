import numpy as np
import pyaudio
import os
from libxheopus import DualOpusEncoder, xOpusDecoder

encoder = DualOpusEncoder("restricted_lowdelay", 48000, "hev2")
encoder.set_bitrates(24000)
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

device_name_output = "Speakers (2- USB Audio DAC   )"
device_index_output = 0
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['name'] == device_name_output:
        device_index_output = dev['index']
        break

streaminput = p.open(format=pyaudio.paInt16, channels=2, rate=48000, input=True, input_device_index=device_index_input)
streamoutput = p.open(format=pyaudio.paInt16, channels=2, rate=48000, output=True, output_device_index=device_index_output)

print(desired_frame_size)

try:
    while True:
        try:
            pcm = np.frombuffer(streaminput.read(desired_frame_size, exception_on_overflow=False), dtype=np.int16)

            if len(pcm) == 0:
                # If PCM is empty, break the loop
                break

            encoded_packets = encoder.encode(pcm)

            print(len(pcm), "-encoded->", len(encoded_packets))


            # print(encoded_packet)
            try:
                decoded_pcm = decoder.decode(encoded_packets)
            except Exception as e:
                decoded_pcm = b""


            # Check if the decoded PCM is empty or not
            if len(decoded_pcm) > 0:
                pcm_to_write = np.frombuffer(decoded_pcm, dtype=np.int16)

                streamoutput.write(pcm_to_write.astype(np.int16).tobytes())
            else:
                print("Decoded PCM is empty")

        except Exception as e:
            print(e)
            raise

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    # Clean up PyAudio streams and terminate PyAudio
    streaminput.stop_stream()
    streaminput.close()
    streamoutput.stop_stream()
    streamoutput.close()
    p.terminate()