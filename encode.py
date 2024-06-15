import sys
import wave
import argparse
from tqdm import tqdm
from libxheopus import DualOpusEncoder, XopusWriter, PSOpusEncoder

# Argument parser setup
parser = argparse.ArgumentParser(description='xHE-Opus Encoder')
parser.add_argument("input", help='Input wav int16 file path or "-" for stdin')
parser.add_argument("output", help='Output xopus file path')
parser.add_argument('-sr', "--samprate", help='Set samples rates', default=48000, type=int)
parser.add_argument('-b', '--bitrate', help='Set Bitrate (bps)', default=64000, type=int)
parser.add_argument('-c', '--compress', help='Set compression: 0-10', default=10, type=int)
parser.add_argument('-l', '--loss', help='Set packet loss: 0-100%', default=0, type=int)
parser.add_argument('-fs', '--framesize', help='Set frame size: 120, 100, 80, 60, 40, 20, 10 or 5', default=120, type=int)
parser.add_argument('-p', '--profile', help='Set xhe-opus profile 1 or 2', default=1, type=int)
parser.add_argument('-bm', '--bitmode', help='Set Bitrate mode: CBR VBR CVBR', default="CVBR")
parser.add_argument('-stm', '--stereomode', help='Set stereo mode (int): mono (0), stereo L-R (1), stereo Mid/Side (2)', default=1, type=int)
parser.add_argument('-bw', '--bandwidth', help='Set bandwidth: auto, fullband, superwideband, wideband, mediumband or narrowband', default="fullband")
parser.add_argument('-a', '--app', help='Set bandwidth: restricted_lowdelay, audio, voip', default="restricted_lowdelay")
parser.add_argument('-v', '--ver', help='Set opus version: hev2 (enable 120, 100, 80 framesize), exper, stable, old', default="hev2")
parser.add_argument('-pred', '--prediction', help='Enable prediction', action='store_true', default=False)
parser.add_argument('-ph', '--phaseinvert', help='Enable phase invert', action='store_true', default=False)
parser.add_argument('-dtx', help='Enable discontinuous transmission', action='store_true', default=False)
parser.add_argument('-sb', "--samebitrate", help='Enable same bitrate (bitrate x2)', action='store_true', default=False)
parser.add_argument('-amn', "--automono", help='Enable auto mono switching in mid/side coding', action='store_true', default=False)
parser.add_argument('-vo', "--voice", help='Enable auto mono switching in mid/side coding', action='store_true', default=False)

args = parser.parse_args()

progress = tqdm()

progress.desc = "creating encoder..."

if args.profile == 1:
    encoder = DualOpusEncoder(args.app, args.samprate, args.ver)
elif args.profile == 2:
    encoder = PSOpusEncoder(args.app, args.samprate, args.ver)
else:
    print("no profile available")
    sys.exit()

if args.profile == 1:
    encoder.set_bitrates(args.bitrate, args.samebitrate)
elif args.profile == 2:
    encoder.set_bitrates(args.bitrate)
    
encoder.set_bitrate_mode(args.bitmode)
encoder.set_bandwidth(args.bandwidth)
encoder.set_compression(args.compress)
encoder.set_packet_loss(args.loss)
encoder.set_feature(args.prediction, args.phaseinvert, args.dtx)
encoder.enable_voice_mode(args.voice)
if args.profile == 1:
    encoder.set_stereo_mode(args.stereomode, args.automono)

desired_frame_size = encoder.set_frame_size(args.framesize)

xopus = XopusWriter(args.output, encoder)

allframe = 0

progress.desc = "reading wav file..."

if args.input == "-":
    wav_file = wave.open(sys.stdin.buffer, 'rb')
else:
    wav_file = wave.open(args.input, 'rb')

while True:
    frames = wav_file.readframes(desired_frame_size)

    if not frames:
        break  # Break the loop when all frames have been read

    allframe += len(frames)

progress.total = allframe
wav_file.rewind()


# Read and process the WAV file in chunks
progress.desc = "encoding..."
while True:
    frames = wav_file.readframes(desired_frame_size)

    if not frames:
        break  # Break the loop when all frames have been read

    xopus.write(frames)

    progress.update(len(frames))

xopus.close()
progress.desc = "encoded"
