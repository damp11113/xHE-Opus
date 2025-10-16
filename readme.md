**im making c++ version for more efficiency!**

![xHE-Opus](https://github.com/damp11113/xHE-Opus/assets/64675096/331bb248-82da-47d2-b930-0caabfd5bc75)
# xHE-Opus
xHE-Opus is extended High Efficiency. It use Dual-Encoder to encode per channel and bitrate is divide 2 for per channel.
It provides optimal coding efficiency with bitrates from 6 to 1020 kilobits per second and above.
It has content loudness metadata on footer.

### xHE-Opus v2
xHE-Opus v2 is extended High Efficiency. It use parametric stereo to encode stereo channel.
It provides optimal coding efficiency with bitrates from 2.5 to 510 kilobits per second and above.

### xHE-Opus v3 (Experiment)
xHE-Opus v3 is Opus with Advanced Parametric stereo coding.

![xheopusquality](https://github.com/damp11113/xHE-Opus/assets/64675096/c2d898c3-7a5e-487b-bd2c-0a364eeec1e7)

# Demo
https://github.com/damp11113/xHE-Opus/assets/64675096/5792f6b4-4ffd-4995-a1ad-97d245b1d1a2

# Install
[PyOgg (damp11113 moded)](https://github.com/damp11113/PyOgg)

# Using
## Gui
easy to use
![image](https://github.com/damp11113/xHE-Opus/assets/64675096/be06bc60-3e51-4866-a8fc-63976a264779)

## Cli
command line with encode.py and player.py
### Encoder
to encode you can use
```bash
$ python3 encode.py
```
```bash
usage: encode.py [-h] [-sr SAMPRATE] [-b BITRATE] [-c COMPRESS] [-l LOSS] [-fs FRAMESIZE] [-bm BITMODE]
                 [-bw BANDWIDTH] [-a APP] [-v VER] [-pred] [-ph] [-dtx] [-sb]
                 input output
encode.py: error: the following arguments are required: input, output
```
simple example
```bash
$ python3 encode.py input.wav output.xopus
```
This will convert to xhe-opus with bitrate 64Kbps (32Kbps per channel), bitrate mode is CVBR, compression is 10 and app is hev2

or if you want to set bitrate you can use `-b <bitrate>` input bit per sec (bps) like
```bash
$ python3 encode.py input.wav output.xopus -b 16000
```

### Decoder/Player
To player or decode this file you can use
```bash
$ python3 player.py input.xopus
```
or if you want only convert to wav you can use
```bash
$ python3 player.py input.xopus -o output.wav
```
## Encode with foobar2000
you can encode xopus with foobar by follow this setup

![image](https://github.com/damp11113/xHE-Opus/assets/64675096/3d285f77-3ac3-4fdf-9320-7de9df30e36c)
