import numpy as np
import scipy as sc
from scipy.io import wavfile
import matplotlib.pyplot as plt


def adaptativeFilter(filter_order, raw_data):
    M = filter_order  # longueur de la fenetre glissante (longeur du X) fs/M pour avoir la plus basse frequence filtrable
    u = raw_data
    step = 1
    e = np.zeros(len(u) - M)
    w = np.zeros(M)
    eps = 0.001
    x_total = []

    for n in range(len(u) - M - 1):
        d = u[n + 1:n + M + 1]
        x = np.flipud(u[n:n + M])
        y = np.dot(x, w)

        x_mag = 1 / (np.dot(x, x) + eps)
        e[n] = d[-1] - y
        w += step * x_mag * x * e[n]

        x_total = np.append(x_total, e[n])

    return np.array(x_total)

speech = sc.io.wavfile.read('test_filtre.wav')[1][:,1]

samplerate = len(speech)
fs = 5000
t = np.linspace(0., 3, samplerate)
amplitude = np.iinfo(np.int16).max/100
data_noise = amplitude * np.sin(2. * np.pi * fs * t)
data = np.add(speech, data_noise)
sc.io.wavfile.write("No_filter.wav", 44100, data)

filt_data = adaptativeFilter(441, data)

sc.io.wavfile.write("filtered.wav", 44100, filt_data)

# ---fft plot--- #
fft_channel_1 = np.fft.fft(data[:22050])
fft_channel_2 = np.fft.fft(filt_data[:22050])

fig, ax = plt.subplots()
freq_ech = 44100
samples = freq_ech//2
xf = np.linspace(0.0, freq_ech/2, int(samples/2))
ax.plot(xf, 2/samples * abs(fft_channel_1[:samples//2]))
ax.set_ylabel('uvolts')
ax.set_xlabel('Hz')
ax.plot(xf, 2/samples * abs(fft_channel_2[:samples//2]))
#ax.plot(np.linspace(0, len(fft_data), samples/2), fft_channels_analysis)
plt.savefig('test_filtre.png')

# f, t, Zxx = sc.signal.spectrogram(fft_channel_1, self.freq_ech, window=('tukey', 0.25), nperseg=1024)
# plt.pcolormesh(t, f, Zxx)
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()