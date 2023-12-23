from datetime import datetime
from pydub import AudioSegment

import gradio as gr
import math
import numpy as np
import os
import pyaudio
import sys
import time
import typing
import vad
import wave

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

class AudioStream():
    FORMAT = pyaudio.paInt16
    # Size of each frame (audio sample), in bytes. If you change FORMAT, make
    # sure this stays up to date!
    FRAME_SZ = 2
    # Frames per second.
    FPS = 16000
    CHANNELS = 1
    def __init__(self):
        pass

    def getSamples(self) -> bytes:
        raise NotImplementedError("getSamples is not implemented!")

class MicStream(AudioStream):
    CHUNK_SZ = 1024

    def __init__(self, which_mic: str, fps: int = AudioStream.FPS):
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.sample_rate = None
        # Each time pyaudio gives us audio data, it's in the form of a chunk of
        # samples. We keep these in a list to keep the audio callback as light
        # as possible. Whenever downstream layers want data, we collapse the
        # list into a single array of data (a bytes object).
        self.chunks = []
        # If set, incoming frames are simply discarded.
        self.paused = False
        self.fps = fps

        print(f"Finding mic {which_mic}", file=sys.stderr)

        got_match = False
        device_index = -1
        if not got_match:
            info = self.p.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            for i in range(0, numdevices):
                if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    device_name = self.p.get_device_info_by_host_api_device_index(0, i).get('name')
                    if which_mic in device_name:
                        print(f"Got matching mic: {device_name}",
                                file=sys.stderr)
                        device_index = i
                        got_match = True
                        break
        if not got_match:
            raise KeyError(f"Mic {which_mic} not found")

        info = self.p.get_device_info_by_host_api_device_index(0, device_index)
        print(f"Found mic {which_mic}: {info['name']}", file=sys.stderr)
        self.sample_rate = int(info['defaultSampleRate'])
        print(f"Mic sample rate: {self.sample_rate}", file=sys.stderr)

        self.stream = self.p.open(
                rate=self.sample_rate,
                channels=self.CHANNELS,
                format=self.FORMAT,
                input=True,
                frames_per_buffer=MicStream.CHUNK_SZ,
                input_device_index=device_index,
                stream_callback=self.onAudioFramesAvailable)

        self.stream.start_stream()

        AudioStream.__init__(self)

    def pause(self, state: bool = True):
        self.paused = state

    def dumpMicDevices(self):
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        for i in range(0, numdevices):
            if (self.p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                device_name = self.p.get_device_info_by_host_api_device_index(0, i).get('name')
                print("Input Device id ", i, " - ", device_name)

    def getMicDevices() -> typing.List[str]:
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        result = []
        for i in range(0, numdevices):
            if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                device_name = p.get_device_info_by_host_api_device_index(0, i).get('name')
                result.append(device_name)
        return result

    def onAudioFramesAvailable(self,
            frames,
            frame_count,
            time_info,
            status_flags):
        if self.paused:
            # Don't literally pause, just start returning silence. This allows
            # the `min_segment_age_s` check to work while paused.
            n_frames = int(frame_count * self.fps /
                    float(self.sample_rate))
            self.chunks.append(np.zeros(n_frames,
                dtype=np.int16).tobytes())
            return (frames, pyaudio.paContinue)

        decimated = b''
        # In pyaudio, a `frame` is a single sample of audio data.
        frame_len = self.FRAME_SZ
        next_frame = 0.0
        # The mic probably has a higher sample rate than Whisper wants, so
        # decrease the sample rate by dropping samples. Note that this
        # algorithm only works if the mic's rate is higher than whisper's
        # expected rate.
        keep_every = float(self.sample_rate) / self.fps
        for i in range(frame_count):
            if i >= next_frame:
                decimated += frames[i*frame_len:(i+1)*frame_len]
                next_frame += keep_every
        self.chunks.append(decimated)

        return (frames, pyaudio.paContinue)

    # Get audio data and the corresponding timestamp.
    def getSamples(self) -> bytes:
        chunks = self.chunks
        self.chunks = []
        result = b''.join(chunks)
        return result

class AudioCollector:
    def __init__(self, stream: AudioStream):
        self.stream = stream
        self.frames = b''
        # Note: by design, this is the only spot where we anchor our timestamps
        # against the real world. This is done to make it possible to profile
        # test cases which read from disk (at much faster than real speed) in
        # the same way that we profile real-time data.
        self.wall_ts = time.time()

    def getAudio(self) -> bytes:
        frames = self.stream.getSamples()
        if frames:
            self.frames += frames
        return self.frames

    def dropAudioPrefix(self, dur_s: float) -> bytes:
        n_bytes = int(dur_s * self.stream.fps) * self.stream.FRAME_SZ
        n_bytes = min(n_bytes, len(self.frames))
        cut_portion = self.frames[:n_bytes]
        self.frames = self.frames[n_bytes:]
        self.wall_ts += float(n_bytes / self.stream.FRAME_SZ) / self.stream.fps
        return cut_portion

    def dropAudioPrefixByFrames(self, dur_frames: int) -> bytes:
        n_bytes = dur_frames * self.stream.FRAME_SZ
        n_bytes = min(n_bytes, len(self.frames))
        cut_portion = self.frames[:n_bytes]
        self.frames = self.frames[n_bytes:]
        self.wall_ts += float(n_bytes / self.stream.FRAME_SZ) / self.stream.fps
        return cut_portion

    def keepLast(self, dur_s: float) -> bytes:
        drop_len = max(0, self.duration() - dur_s)
        return self.dropAudioPrefix(drop_len)

    def dropAudio(self):
        self.wall_ts += self.duration()
        cut_portion = self.frames
        self.frames = b''
        return cut_portion

    def duration(self):
        return len(self.frames) / (self.stream.fps * self.stream.FRAME_SZ)

    def begin(self):
        return self.wall_ts

    def now(self):
        return self.begin() + self.duration()

class AudioCollectorFilter:
    def __init__(self, parent: AudioCollector):
        self.parent = parent
        self.stream = self.parent.stream

    def getAudio(self) -> bytes:
        return self.parent.getAudio()
    def dropAudioPrefix(self, dur_s: float):
        return self.parent.dropAudioPrefix(dur_s)
    def dropAudioPrefixByFrames(self, dur_frames: int):
        return self.parent.dropAudioPrefixByFrames(dur_frames)
    def keepLast(self, dur_s):
        return self.parent.keepLast(dur_s)
    def dropAudio(self):
        return self.parent.dropAudio()
    def duration(self):
        return self.parent.duration()
    def begin(self):
        return self.parent.begin()
    def now(self):
        return self.parent.now()

class NormalizingAudioCollector(AudioCollectorFilter):
    def __init__(self, parent: AudioCollector):
        AudioCollectorFilter.__init__(self, parent)

    def getAudio(self) -> bytes:
        audio = self.parent.getAudio()

        audio = AudioSegment(audio, sample_width=self.stream.FRAME_SZ,
                frame_rate=self.stream.fps, channels=self.stream.CHANNELS)
        audio = audio.normalize()

        frames = np.array(audio.get_array_of_samples())
        frames = np.int16(frames).tobytes()

        return frames

class CompressingAudioCollector(AudioCollectorFilter):
    def __init__(self, parent: AudioCollector):
        AudioCollectorFilter.__init__(self, parent)

    def getAudio(self) -> bytes:
        audio = self.parent.getAudio()

        audio = AudioSegment(audio,
                sample_width=self.stream.FRAME_SZ,
                frame_rate=self.stream.fps,
                channels=self.stream.CHANNELS)
        # subtle compression has a slight positive effect on my benchmark
        audio = audio.compress_dynamic_range(threshold=-10, ratio=2.0)

        frames = np.array(audio.get_array_of_samples())
        frames = np.int16(frames).tobytes()

        return frames

class AudioSegmenter:
    def __init__(self,
            min_silence_ms=250,
            max_speech_s=5,
            stream: AudioStream = None):
        self.vad_options = vad.VadOptions(
                min_silence_duration_ms=min_silence_ms,
                max_speech_duration_s=max_speech_s)
        self.stream = stream
        pass

    def segmentAudio(self, audio: bytes):
        audio = np.frombuffer(audio,
                dtype=np.int16).flatten().astype(np.float32) / 32768.0
        return vad.get_speech_timestamps(audio, vad_options=self.vad_options)

    # Returns the stable cutoff (if any) and whether there are any segments.
    def getStableCutoff(self, audio: bytes) -> typing.Tuple[int, bool]:
        min_delta_frames = int((self.vad_options.min_silence_duration_ms *
                self.stream.fps) / 1000)
        cutoff = None

        last_end = None
        segments = self.segmentAudio(audio)

        for i in range(len(segments)):
            s = segments[i]
            #print(f"s: {s}")
            #print(f"last_end: {last_end}")

            if last_end:
                delta_frames = s['start'] - last_end
                #print(f"delta frames: {delta_frames}")
                if delta_frames > min_delta_frames:
                    cutoff = s['start']
            else:
                last_end = s['end']

            if i == len(segments) - 1:
                now = int(len(audio) / self.stream.FRAME_SZ)
                delta_frames = now - s['end']
                if delta_frames > min_delta_frames:
                    cutoff = now - int(min_delta_frames / 2)

        return (cutoff, len(segments) > 0)

def install_in_venv(pkgs: typing.List[str]) -> bool:
    pkgs_str = " ".join(pkgs)
    print(f"Installing {pkgs_str}")
    pip_proc = subprocess.Popen(
            f"Resources/Python/python.exe -m pip install {pkgs_str} --no-warn-script-location".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
    pip_stdout, pip_stderr = pip_proc.communicate()
    pip_stdout = pip_stdout.decode("utf-8")
    pip_stderr = pip_stderr.decode("utf-8")
    print(pip_stdout, file=sys.stderr)
    print(pip_stderr, file=sys.stderr)
    if pip_proc.returncode != 0:
        print(f"`pip install {pkgs_str}` exited with {pip_proc.returncode}",
                file=sys.stderr)
        return False
    return True

def saveAudio(audio: bytes, path: str, stream: AudioStream):
    with wave.open(path, 'wb') as wf:
        print(f"Saving audio to {path}", file=sys.stderr)
        wf.setnchannels(stream.CHANNELS)
        wf.setsampwidth(stream.FRAME_SZ)
        wf.setframerate(stream.fps)
        wf.writeframes(audio)

def concatenateWavFiles(output_path):
    # List all .wav files in the CWD
    wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]

    # Initialize parameters for wave file
    params = None

    # Open the output file
    with wave.open(output_path, 'wb') as output_wav:
        for wav_file in wav_files:
            if os.path.abspath(wav_file) == os.path.abspath(output_path):
                print(f"Skip adding output file ({wav_file}) to itself")
                continue
            print(f"Processing {wav_file}")
            with wave.open(wav_file, 'rb') as input_wav:
                # Check if parameters are the same for each file
                if params is None:
                    params = input_wav.getparams()
                    output_wav.setparams(params)

                # Read and write frames
                frames = input_wav.readframes(input_wav.getnframes())
                output_wav.writeframes(frames)

class AppControl:
    run = True
app_ctrl = AppControl()

def recordMeDaddy(
        mic_device: str,
        min_volume: float = -1.3,
        max_volume: float = -0.8
        ):
    app_ctrl.run = True

    stream = MicStream(mic_device)
    stream_hd = MicStream(mic_device, fps=44100)

    collector = AudioCollector(stream)
    #collector = NormalizingAudioCollector(collector)
    collector = CompressingAudioCollector(collector)

    collector_hd = AudioCollector(stream_hd)
    #collector_hd = NormalizingAudioCollector(collector_hd)
    collector_hd = CompressingAudioCollector(collector_hd)

    min_silence_ms = 1000
    max_speech_s = 30
    segmenter = AudioSegmenter(
            min_silence_ms=min_silence_ms,
            max_speech_s=max_speech_s,
            stream=stream)

    while app_ctrl.run:
        audio = collector.getAudio()
        collector_hd.getAudio()
        stable_cutoff, has_audio = segmenter.getStableCutoff(audio)

        #print(f"has audio: {has_audio}")
        #print(f"stable cutoff: {stable_cutoff}")

        if has_audio and stable_cutoff:
            commit_audio = collector.dropAudioPrefixByFrames(stable_cutoff)
            print(f"stable cutoff: {stable_cutoff}")
            hd_cutoff = int(math.floor(stable_cutoff * stream_hd.fps /
                stream.fps))
            print(f"hd cutoff: {hd_cutoff}")
            commit_audio_hd = collector_hd.dropAudioPrefixByFrames(hd_cutoff)
            print(f"hd audio len: {len(commit_audio_hd)}")

            # Calculate naive measure of volume
            audio_v = AudioSegment(commit_audio_hd,
                    sample_width=stream_hd.FRAME_SZ,
                    frame_rate=stream_hd.fps,
                    channels=stream_hd.CHANNELS)
            audio_v = np.array(audio_v.get_array_of_samples())
            audio_v = np.int16(audio_v)
            audio_v = np.sqrt(np.mean(np.square(audio_v)))
            audio_v /= np.sqrt(len(commit_audio_hd) / stream_hd.FRAME_SZ)
            audio_v = math.log(audio_v, 10)
            print(f"volume: {audio_v}")
            # cutoff is a fine-tuned value based on volumes seen while in vr
            # (index mic)
            if audio_v < min_volume or audio_v > max_volume:
                # Discard sample
                print("Discarding too-quiet/too-loud segment")
                collector.keepLast(1.0)
                collector_hd.keepLast(1.0)
                continue


            ts = datetime.fromtimestamp(time.time())
            filename = str(ts.strftime('%Y_%m_%d__%H-%M-%S')) + ".wav"
            saveAudio(commit_audio_hd, filename, stream_hd)

        if not has_audio:
            #print("VAD detects no audio, skip transcription", file=sys.stderr)
            collector.keepLast(1.0)
            collector_hd.keepLast(1.0)
    print("Stopped recording")

def getOutput() -> str:
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()

def stopApp():
    print("Requesting app stop")
    app_ctrl.run = False

if __name__ == "__main__":
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    sys.stdout = Logger("output.log")

    print(f"Set cwd to {os.getcwd()}", file=sys.stderr)

    with gr.Blocks() as demo:
        mic_choices = MicStream.getMicDevices()
        mic_device = gr.Dropdown(choices=mic_choices, label="Microphone")
        min_volume = gr.Number(label="Minimum volume", value=-1.3)
        max_volume = gr.Number(label="Maximum volume", value=-0.8)
        record_audio = gr.Button("Record audio")
        stop_recording = gr.Button("Stop recording")
        concatenated_path = gr.Text(label="Combined audio filename", value="combined.wav")
        min_length = gr.Number(label="Minimum length (seconds)", value=3.0)
        concatenate_audio = gr.Button("Combine audio files")

        dbg_output = gr.Text(label="Output")

        record_audio.click(recordMeDaddy, [mic_device, min_volume, max_volume],
                dbg_output)
        stop_recording.click(stopApp, [], dbg_output)
        concatenate_audio.click(concatenateWavFiles, [concatenated_path],
                dbg_output)

        demo.load(getOutput, None, dbg_output, every=0.5)
    demo.launch()
    sys.exit(0)

    concatenateWavFiles("concatenated.wav")
    sys.exit(0)

