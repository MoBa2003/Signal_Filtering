import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def generate_random_signals(length=1000, distribution='normal'):
        """
        Generate random signals with different distributions
        
        Args:
            length (int): Length of signals
            distribution (str): Type of distribution ('normal', 'uniform', 'exponential')
        
        Returns:
            tuple: Two random signals
        """
        if distribution == 'normal':
            signal1 = np.random.normal(0, 1, length)
            signal2 = np.random.normal(0, 1, length)
            # signal2 = np.zeros(length)
        elif distribution == 'uniform':
            signal1 = np.random.uniform(-1, 1, length)
            signal2 = np.random.uniform(-1, 1, length)
        elif distribution == 'exponential':
            signal1 = np.random.exponential(1, length)
            signal2 = np.random.exponential(1,)
        
        return signal1, signal2

def generate_pulse_signal(length, pulse_start, pulse_end, amplitude=1):
    """
    تولید سیگنال پالس
    
    پارامترها:
    length: طول کل سیگنال (تعداد نمونه‌ها)
    pulse_start: نقطه شروع پالس
    pulse_end: نقطه پایان پالس
    amplitude: دامنه پالس (پیش‌فرض 1)
    
    خروجی:
    سیگنال پالس به صورت یک آرایه numpy
    """
    # ایجاد سیگنال صفر
    signal = np.zeros(length)
    
    # تنظیم مقادیر پالس در بازه مشخص
    signal[pulse_start:pulse_end] = amplitude
    
    return signal


# # تولید دو سیگنال تصادفی
# np.random.seed(42)  # برای تولید مقادیر قابل تکرار
# signal1 = np.random.rand(500)  # سیگنال تصادفی اول
# signal2 = np.random.rand(500)  # سیگنال تصادفی دوم
signal1,signal2 = generate_random_signals()
# signal1 = generate_pulse_signal(1000,0,100)
# signal2 = generate_pulse_signal(1000,0,300)

# High-pass filter design





# محاسبه کانولوشن سیگنال‌ها
convolved_signal = np.convolve(signal1, signal2,mode='full')

# تحلیل سیگنال‌ها در حوزه فرکانس
fft_signal1 = np.fft.fft(signal1)
fft_signal2 = np.fft.fft(signal2)
fft_convolved = np.fft.fft(convolved_signal)

frequencies = np.fft.fftfreq(len(signal1))

# رسم سیگنال‌ها در حوزه زمان
plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.plot(signal1)
plt.title("Signal 1 (Time Domain)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.subplot(3, 2, 2)
plt.plot(np.abs(fft_signal1))
plt.title("Signal 1 (Frequency Domain)")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.subplot(3, 2, 3)
plt.plot(signal2)
plt.title("Signal 2 (Time Domain)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.subplot(3, 2, 4)
plt.plot(np.abs(fft_signal2))
plt.title("Signal 2 (Frequency Domain)")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.subplot(3, 2, 5)
plt.plot(convolved_signal)
plt.title("Convolved Signal (Time Domain)")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.subplot(3, 2, 6)
plt.plot(np.abs(fft_convolved))
plt.title("Convolved Signal (Frequency Domain)")
plt.xlabel("Frequency")
plt.ylabel("Magnitude")

plt.tight_layout()
plt.show()
