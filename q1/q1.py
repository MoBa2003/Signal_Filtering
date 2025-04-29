import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as signal

# 1. Load the MP3 audio file
mp3_file = './q1/aud1.mp3'
audio_signal, sample_rate = sf.read(mp3_file)

# Ensure the signal is mono (convert stereo to mono if needed)
if audio_signal.ndim > 1:
    audio_signal = np.mean(audio_signal, axis=1)  # Convert stereo to mono

# Normalize the signal to the range [-1, 1]
audio_signal = audio_signal / np.max(np.abs(audio_signal))

# Time axis for plotting
time_axis = np.linspace(0, len(audio_signal) / sample_rate, num=len(audio_signal))

# Function to compute FFT
def compute_fft(signal, sample_rate):
    fft_signal = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1 / sample_rate)
    magnitude = np.abs(fft_signal)
    return frequencies[:len(frequencies) // 2], magnitude[:len(magnitude) // 2]

# Compute FFT for the original signal
frequencies, magnitude = compute_fft(audio_signal, sample_rate)

# 3. Design digital filters
nyquist_freq = sample_rate / 2

# Low-pass filter design
low_cutoff_freq = 2000  # Low-pass cutoff frequency in Hz
normalized_low_cutoff = low_cutoff_freq / nyquist_freq
b_lp, a_lp = signal.butter(4, normalized_low_cutoff, btype='low', analog=False)

# High-pass filter design
high_cutoff_freq = 1000  # High-pass cutoff frequency in Hz
normalized_high_cutoff = high_cutoff_freq / nyquist_freq
b_hp, a_hp = signal.butter(4, normalized_high_cutoff, btype='high', analog=False)

# 4. Apply filters
filtered_lp = signal.filtfilt(b_lp, a_lp, audio_signal)
filtered_hp = signal.filtfilt(b_hp, a_hp, audio_signal)
filtered_both = signal.filtfilt(b_hp, a_hp, filtered_lp)

# Compute FFT for filtered signals
freq_lp, mag_lp = compute_fft(filtered_lp, sample_rate)
freq_hp, mag_hp = compute_fft(filtered_hp, sample_rate)
freq_both, mag_both = compute_fft(filtered_both, sample_rate)

# Plot results
plt.figure(figsize=(18, 16))

signals = [
    ("Original Signal", audio_signal, frequencies, magnitude),
    ("Low-pass Filtered Signal", filtered_lp, freq_lp, mag_lp),
    ("High-pass Filtered Signal", filtered_hp, freq_hp, mag_hp),
    ("Both Filtered Signal", filtered_both, freq_both, mag_both),
]

for i, (title, signal_time, freq, mag) in enumerate(signals):
    # Time domain plot
    plt.subplot(4, 2, 2 * i + 1)
    plt.plot(time_axis, signal_time, label=f'{title} (Time Domain)')
    plt.title(f'{title} (Time Domain)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.legend()

    # Frequency domain plot
    plt.subplot(4, 2, 2 * i + 2)
    plt.plot(freq, mag, label=f'{title} (Frequency Domain)')
    plt.title(f'{title} (Frequency Domain)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.legend()

plt.tight_layout()
plt.show()
