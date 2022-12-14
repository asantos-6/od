import IPython.display as ipd
import librosa
import librosa.display
from madmom.audio import Spectrogram
import matplotlib.pyplot as plt


def read_audio(file_path, trim_interval=None, print_it=False):
    audio, sr = librosa.load(file_path)
    if trim_interval is not None:
        ti = trim_interval[0]
        tf = trim_interval[1]
        audio = trim_audio(audio, sr, ti, tf)
    if print_it:
        print(audio.shape)
        print(sr)
    return audio, sr


def trim_audio(audio, sr, ti, tf):
    i = ti * sr
    f = tf * sr
    t_audio = audio[i:f]
    return t_audio


def plot_audio(x, sr):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(x, sr=sr)
    return


def play_audio(file_path, sr):
    ipd.Audio(file_path, rate=sr)  # load a local WAV file


    # Function that calculates the total seconds of an audio file
def audio_seconds(audio, sr):
    samples = audio.shape[0]
    s = samples / sr
    return s


def show_spectrogram(x, sr, y_scale='linear', frame_size=2048):
    X = librosa.stft(x, n_fft=frame_size)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis=y_scale, n_fft=frame_size)
    return


def madmom_spectrogram(file_path):
    spec = Spectrogram(file_path)
    return spec
