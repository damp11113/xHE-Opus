import dearpygui.dearpygui as dpg
import easygui
import threading
import libxheopus
import os
import subprocess
import tempfile
import wave
import pyaudio
import ctypes

class App:
    def __init__(self):
        self.inputfilepath = None
        self.outputpath = None

        self.deinputfilepath = None
        self.deoutputpath = None
        self.derender = None
        self.delen = 0
        self.depausepos = 0
        self.decurrentplay = 0
        self.deplay = False
        self.deopened = False
        self.deplayframeskip = 0

        portaudio = pyaudio.PyAudio()
        self.decoder = None

        self.streamoutput = portaudio.open(format=pyaudio.paInt16, channels=2, rate=48000, output=True)

    def selectinputfile(self, sender, data):
        file_path = easygui.fileopenbox("Select Video/Audio File")
        dpg.set_value("inpathshow", f"input: {file_path}")
        self.inputfilepath = file_path

    def selectoutputpath(self, sender, data):
        file_path = easygui.diropenbox(title="Select Output Folder")
        dpg.set_value("outpathshow", f"output: {file_path}")
        self.outputpath = file_path

    def changeversionopus(self, sender, data):
        if data == "hev2":
            dpg.configure_item("opusframesize", items=["120", "100", "80", "60", "40", "20", "10", "5"])
        else:
            dpg.configure_item("opusframesize", items=["60", "40", "20", "10", "5"])
            if int(dpg.get_value("opusframesize")) > 60:
                dpg.configure_item("opusframesize", default_value="60")

    def selectdeoutputpath(self, sender, data):
        file_path = easygui.diropenbox()
        dpg.set_value("deoutpathshow", f"output: {file_path}")
        self.deoutputpath = file_path
        if file_path != None or file_path != "" and self.deinputfilepath != None or self.deinputfilepath != "":
            dpg.configure_item("deplayconvert", show=True)
        else:
            dpg.configure_item("deplayconvert", show=False)

    def selectdeinputfile(self, sender, data):
        file_path = easygui.fileopenbox("Select Xopus File", filetypes=["*.xopus"], default="*.xopus")
        dpg.set_value("deinpathshow", f"input: {file_path}")

        self.deinputfilepath = file_path

        if file_path != None or file_path != "None" or file_path != "":
            if self.deopened:
                self.derender.close()
            self.derender = libxheopus.XopusReader(file_path)
            self.deopened = True
            self.stopaudio(None, None)
            self.decurrentplay = 0
            self.depausepos = 0
            self.deplayframeskip = 0
            self.delen = 0
            dpg.configure_item("deinfo", show=True)
            thread = threading.Thread(target=self.readmetadatathread, daemon=True)
            thread.start()
        else:
            dpg.configure_item("deinfo", show=False)
            dpg.configure_item("deplayconvert", show=False)

    def convert(self):
        try:
            total = 0
            current = 0
            filename = os.path.splitext(os.path.basename(self.inputfilepath))[0]

            dpg.set_value("convertstatus", "init encoder...")
            encoder = libxheopus.DualOpusEncoder(dpg.get_value("opusapp"), 48000, dpg.get_value("opusversion"))
            encoder.set_bitrate_mode(dpg.get_value("opusbitmode"))
            encoder.set_bandwidth(dpg.get_value("opusbandwidth"))
            encoder.set_bitrates(int(dpg.get_value("opusbitrate")*1000))
            encoder.set_compression(dpg.get_value("opuscompression"))
            encoder.set_packet_loss(dpg.get_value("opuspacketloss"))
            encoder.set_feature(dpg.get_value("opusenapred"), False, dpg.get_value("opusenadtx"))
            desired_frame_size = encoder.set_frame_size(int(dpg.get_value("opusframesize")))

            dpg.set_value("convertstatus", "init writer...")
            writer = libxheopus.XopusWriter(f"{self.outputpath}/{filename}.xopus", encoder)

            dpg.set_value("convertstatus", "converting to wav int16 with ffmpeg...")
            temp_dir = tempfile.mkdtemp()
            temp_wave_file = os.path.join(temp_dir, filename + ".wav")

            subprocess.run(["ffmpeg", "-i", self.inputfilepath, "-vn", "-acodec", "pcm_s16le", "-ac", "2", "-ar", "48000", temp_wave_file], check=True)

            dpg.set_value("convertstatus", "reading temp wav...")

            wav_file = wave.open(temp_wave_file, 'rb')
            while True:
                frames = wav_file.readframes(desired_frame_size)

                if not frames:
                    break  # Break the loop when all frames have been read

                total += len(frames)

            wav_file.rewind()

            dpg.set_value("convertstatus", "Encoding...")
            while True:
                frames = wav_file.readframes(desired_frame_size)

                if not frames:
                    break  # Break the loop when all frames have been read

                writer.write(frames)

                current += len(frames)

                dpg.set_value("convertprogbar", min(1.0, max(0.0, current / total))) # show percentage

            writer.close()
            wav_file.close()
            os.remove(temp_wave_file)

        except Exception as e:
            dpg.set_value("convertstatus", str(e))
        else:
            dpg.set_value("convertstatus", "Converted")

        dpg.configure_item("okconvertbutton", show=True)

    def startconvert(self, sender, data):
        dpg.configure_item("okconvertbutton", show=False)
        dpg.configure_item("convertingwindow", show=True)
        if self.outputpath is None or self.outputpath == "" or self.inputfilepath is None or self.inputfilepath == "":
            dpg.set_value("convertstatus", "Please check input file and output file")
            dpg.configure_item("okconvertbutton", show=True)
        else:
            thread = threading.Thread(target=self.convert, daemon=True)
            thread.start()

    def readmetadatathread(self):
        metadata = self.derender.readmetadata()
        dpg.set_value("deloudness", f'Loudness: {int(metadata["footer"]["contentloudness"])} DBFS')
        dpg.set_value("demetadata", f'Metadata: {metadata["header"]}')
        self.delen = metadata["footer"]["length"]

    def playaudiothread(self):
        self.decoder = libxheopus.DualOpusDecoder()
        for data in self.derender.decode(self.decoder, True, self.depausepos):
            if self.deplay:

                if data != b"":
                    self.streamoutput.write(data)
                else:
                    self.deplayframeskip += 1
                    dpg.set_value("destatusfs", f"Frame Skip: {self.deplayframeskip}")

                self.decurrentplay += 1

                dpg.set_value("deplayingprog", min(1.0, max(0.0, self.decurrentplay / self.delen)))
            else:
                if self.decurrentplay != 0:
                    self.depausepos = self.decurrentplay

                break

        if dpg.get_value("deplayingprog") != 1:
            dpg.set_value("destatus", "Paused")
        else:
            dpg.set_value("destatus", "Stopped")

    def playpauseaudio(self, sender, data):
        dpg.configure_item("destatusfs", show=True)
        self.deplay = not self.deplay

        if self.deplay:
            if self.depausepos != 0:
                self.decurrentplay = self.depausepos

            dpg.set_value("destatus", "Playing")
            dpg.configure_item("deplaybutton", label="Pause")
            thread = threading.Thread(target=self.playaudiothread, daemon=True)
            thread.start()
        else:
            dpg.set_value("destatus", "Paused")
            dpg.configure_item("deplaybutton", label="Play")

    def stopaudio(self, sender, data):
        dpg.configure_item("destatusfs", show=False)
        dpg.set_value("destatusfs", "Frame Skip: 0")
        self.decurrentplay = 0
        self.depausepos = 0
        self.deplayframeskip = 0
        self.deplay = False
        dpg.set_value("deplayingprog", 0)
        dpg.set_value("destatus", "Stopped")
        dpg.configure_item("deplaybutton", label="Play")

    def deconvertthread(self):
        outwav = wave.open(self.deoutputpath + "/" + os.path.splitext(os.path.basename(self.deinputfilepath))[0] + ".wav", "w")
        # Set the parameters of the WAV file
        outwav.setnchannels(2)  # Stereo
        outwav.setsampwidth(2)  # 2 bytes (16 bits) per sample
        outwav.setframerate(48000)

        self.decoder = libxheopus.DualOpusDecoder()
        for data in self.derender.decode(self.decoder, True, self.depausepos):
            self.decurrentplay += 1

            # Write the audio data to the file
            if data != b"":
                outwav.writeframes(data)
            else:
                self.deplayframeskip += 1
                dpg.set_value("destatusfs", f"Frame Skip: {self.deplayframeskip}")

            dpg.set_value("deplayingprog", min(1.0, max(0.0, self.decurrentplay / self.delen)))

        outwav.close()
        self.decurrentplay = 0
        dpg.set_value("destatus", "Converted")
        dpg.configure_item("destatusfs", show=False)

    def startdeconvert(self, sender, data):
        self.stopaudio(None, None)
        dpg.configure_item("destatusfs", show=True)
        dpg.set_value("destatus", "Converting")
        thread = threading.Thread(target=self.deconvertthread, daemon=True)
        thread.start()

    def window(self):
        with dpg.window(label="Encoder", width=420, no_close=True):
            dpg.add_text("input:", tag="inpathshow")
            dpg.add_text("output:", tag="outpathshow")
            dpg.add_button(label="Select Input File", callback=self.selectinputfile)
            dpg.add_button(label="Select Output Path", callback=self.selectoutputpath)
            dpg.add_combo(["hev2", "exper", "stable", "old"], label="Version", default_value="hev2", tag="opusversion", callback=self.changeversionopus)
            dpg.add_combo(["120", "100", "80", "60", "40", "20", "10", "5"], label="Frame Size (ms)", tag="opusframesize", default_value="120")
            dpg.add_combo(["voip", "audio", "restricted_lowdelay"], label="Application", default_value="restricted_lowdelay", tag="opusapp")
            dpg.add_combo(["VBR", "CVBR", "CBR"], label="Bitrate Mode", default_value="CVBR", tag="opusbitmode")
            dpg.add_combo(["auto", "fullband", "superwideband", "wideband", "mediumband", "narrowband"], label="Bandwidth", tag="opusbandwidth", default_value="fullband")
            dpg.add_input_float(label="Bitrates", min_value=5, max_value=1020, min_clamped=True, max_clamped=True, step_fast=1, default_value=64, tag="opusbitrate")
            dpg.add_input_int(label="Compression Level", max_clamped=True, min_clamped=True, min_value=0, max_value=10, default_value=10, tag="opuscompression")
            dpg.add_input_int(label="Packet Loss", max_clamped=True, min_clamped=True, min_value=0, max_value=100, default_value=0, tag="opuspacketloss")
            dpg.add_checkbox(label="Prediction", tag="opusenapred")
            dpg.add_checkbox(label="DTX", tag="opusenadtx")
            dpg.add_button(label="Convert", callback=self.startconvert)

        with dpg.window(label="converting", show=False, tag="convertingwindow", modal=True, no_resize=True, no_move=True, no_title_bar=True, width=320):
            dpg.add_text("converting...", tag="convertstatus")
            dpg.add_progress_bar(tag="convertprogbar")
            dpg.add_button(label="OK", callback=lambda: dpg.configure_item("convertingwindow", show=False), tag="okconvertbutton", show=False)

        with dpg.window(label="Player/Decoder", height=320, width=420, pos=(500, 0), no_close=True):
            dpg.add_text("input:", tag="deinpathshow")
            dpg.add_text("output:", tag="deoutpathshow")
            dpg.add_button(label="Select Input File", callback=self.selectdeinputfile)
            dpg.add_button(label="Select Output Path", callback=self.selectdeoutputpath)
            with dpg.group(tag="deinfo", show=False):
                dpg.add_text("Loudness: ? DBFS", tag="deloudness")
                dpg.add_text("Metadata: ?", tag="demetadata", wrap=400)
                dpg.add_progress_bar(tag="deplayingprog")
                dpg.add_text("Frame Skip: 0", tag="destatusfs", show=False)
                dpg.add_text("Stopped", tag="destatus")
                dpg.add_button(label="Play", tag="deplaybutton", callback=self.playpauseaudio)
                dpg.add_button(label="Stop", tag="destopbutton", callback=self.stopaudio)
                dpg.add_button(label="Convert", tag="deplayconvert", show=False, callback=self.startdeconvert)

    def init(self):
        dpg.create_context()
        dpg.create_viewport(title='xHE-Opus GUI', width=1280, height=720)  # set viewport window
        dpg.setup_dearpygui()
        # -------------- add code here --------------
        self.window()

        with dpg.texture_registry():
            width, height, channels, data = dpg.load_image("xHE-Opus.png")

            dpg.add_static_texture(width=512, height=192, default_value=data, tag="app_logo_background")

        with dpg.window(no_background=True, no_title_bar=True, no_move=True, no_resize=True, tag="backgroundviewportlogo"):
            dpg.add_image("app_logo_background")
            dpg.add_text("ThaiSDR Solutions", pos=(230, 230))
        # -------------------------------------------
        dpg.show_viewport()

        ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

        while dpg.is_dearpygui_running():
            self.render()
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

    def render(self):
        # insert here any code you would like to run in the render loop
        # you can manually stop by using stop_dearpygui() or self.exit()
        dpg.configure_item("backgroundviewportlogo", pos=(dpg.get_viewport_width() - 550, dpg.get_viewport_height() - 300))

    def exit(self):
        dpg.destroy_context()


app = App()
app.init()