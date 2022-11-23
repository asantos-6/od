import numpy as np
import matplotlib.pyplot as plt

from onset_detection.od import calculate_onset_times
from onset_detection.audio import plot_audio


def plot_df(audio, sr, df_bins, df, c='r', a=0.5, overlap=False):
    samples = audio.shape[0]
    s = samples / sr
    times = np.linspace(0, s, df_bins)
    onset_times = calculate_onset_times(df, df_bins, s)
    if not overlap:
        plt.figure(figsize=(14, 5))
    if df.dtype == int:
        plt.vlines(onset_times, min(audio), max(audio), color=c, alpha=a)
    else:
        plt.plot(times, df, color=c, alpha=a)
    return onset_times


# Plot audio and DF combined
def plot_combined(audio, sr, df_bins, df):
    plot_audio(audio, sr)
    onset_times = plot_df(audio, sr, df_bins, df, c='r', a=0.5, overlap=True)
    return onset_times
