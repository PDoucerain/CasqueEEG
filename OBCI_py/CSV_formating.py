import numpy as np
import pandas as pd
import scipy as sc
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as pyplot
import sklearn
from sklearn import linear_model
from matplotlib import style
import pickle
import csv

labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Status']
OV_labels = ['Time:128Hz', 'Epoch', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Status', 'Ref_Nose', 'Event Id', 'Event Date', 'Event Duration']

# ---Importation des donnees du .mat--- #
data_filename = 'U001ai.mat'
data_folder = 'SSVEP_Training/Dataset/'
data_path = data_folder + data_filename
data = loadmat(data_path)
datacsv = data['eeg']
events = data['events']

# ---Echantillonage--- #
freq_ech = 128
exp_time = 5
timestep = 1/freq_ech
samples = freq_ech*exp_time
ep_lgt = exp_time*freq_ech
ref_nose = 4179.179
PI = 3.1416

# ---Frequences utilisees par SSVEP--- #
f_SSVEP = [12.00, 10.00, 8.57, 7.50, 6.66]
bp_freq = [f_SSVEP[-1], f_SSVEP[0]]
bp_wn = [i * 2.0/freq_ech for i in bp_freq]         # for butterworth filter

datacsv = list(datacsv)
datacsv = np.transpose(datacsv)

nb_channels = len(datacsv[0])
nb_epoch = len(datacsv)//ep_lgt
print("nb_channels :", nb_channels)
print("nb_epoch :", nb_epoch)

# ---Ajout des labels des données--- #
df = pd.DataFrame(data=datacsv, columns=labels)
timestamp = np.linspace(0, len(datacsv)/freq_ech, len(datacsv))

# ---Epoch definition--- #
epoch_label = []
j = 0
for ii in range(int(len(datacsv)/(5*freq_ech))+1):
    for jj in range(exp_time*freq_ech):
        epoch_label.append(j)
        if len(epoch_label) == len(datacsv):
            break
    j += 1

fft_data = np.array(datacsv)
print("fftdata :", len(datacsv))
# ---Soustraction de la valeur neutre aux donnees--- #
fft_data = np.subtract(fft_data, ref_nose)

# ---Definition d'une epoch--- #
epoch = nb_epoch * [fft_data[0:ep_lgt, 1]]

# ---Definition d'un channel--- #
channel = nb_channels*[None]
for i in range(nb_channels):
    channel[i] = epoch[:]

print("longueur channel :", np.shape(channel))

# ---Integration des donnees dans array 3D [channel][epoch][echantillon]--- #
for c in range(nb_channels):
    for e in range(1, nb_epoch):
        channel[c][e] = fft_data[((ep_lgt*e)-ep_lgt):(ep_lgt*e), c]

# ---Choix du channel et de l'epoch--- #
chan = 6
epo = 0

# ---Filtrage passe-bande des donnees--- #
[b, a] = sc.signal.butter(5, bp_wn, btype='bandpass')
fft_channel_1 = signal.filtfilt(b, a, channel[chan][epo])
#fft_channel_1 = channel[5][3]
fft_channel_1 = np.fft.fft(fft_channel_1)

#print(channel[0][8])
fig, ax = pyplot.subplots()
xf = np.linspace(0.0, freq_ech/2, int(samples/2))
ax.plot(xf, 2/samples * abs(fft_channel_1[:samples//2]))
ax.set_ylabel('uvolts')
ax.set_xlabel('Hz')
ax.set_title('Channel eeg {}, epoch : {}'.format(labels[chan], epo+1))
#ax.plot(np.linspace(0, len(fft_data), freq_ech*5), fft_data)
pyplot.show()

# ---Detection de la frequence predominante--- #
fft_channels = 2/samples * abs(fft_channel_1[:samples//2])
#max_freq = [i for i, x in enumerate(fft_channels[5:]) if x == max(fft_channels[5:])]
max_freq = np.argsort(fft_channels)
max_freq = max_freq[::-1]

# Ajout des colonnes manquantes au CSV
df['Time:128Hz'] = timestamp
df['Epoch'] = epoch_label
df['Event Id'] = 0
df['Event Date'] = 0
df['Event Duration'] = 0
df['Ref_Nose'] = 4179.179

# Reorganisation des colonnes dans l'ordre d'OpenVibe
df = df[OV_labels]

# Conversion de la liste vers CSV
df.to_csv(data_filename, index=False, header=True)
