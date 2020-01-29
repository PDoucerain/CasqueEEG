import numpy as np
import pandas as pd
import scipy as sc
from scipy.io import loadmat
from scipy import signal
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle
import csv

labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Status']
OV_labels = ['Time:128Hz', 'Epoch', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Status', 'Ref_Nose', 'Event Id', 'Event Date', 'Event Duration']

# ---Importation des donnees du .mat--- #
data = loadmat("SSVEP_Training/Dataset/U001ai.mat")
datacsv = data['eeg']
events = data['events']

# ---Echantillonage--- #
freq_ech = 128
exp_time = 5
timestep = 1/freq_ech
samples = freq_ech*exp_time
ep_lgt = exp_time*freq_ech
ref_nose = 4179.179

# ---Frequences utilisees par SSVEP--- #
f_SSVEP = [12.00, 10.00, 8.57, 7.50, 6.66]
#wn = (1/(freq_ech/2)) * f_SSVEP    #Freq normalise 0 a nyquist


datacsv = list(datacsv)
datacsv = np.transpose(datacsv)

nb_channels = len(datacsv[0])
nb_epoch = len(datacsv)//ep_lgt
print("nb_channels :", nb_channels)
print("nb_epoch :", nb_epoch)
#Ajout des labels des données
df = pd.DataFrame(data=datacsv, columns=labels)

timestamp = np.linspace(0, len(datacsv)/freq_ech, len(datacsv))

# Epoch definition
epoch = []
j = 0
for ii in range(int(len(datacsv)/(5*freq_ech))+1):
    for jj in range(5*freq_ech):
        epoch.append(j)
        if len(epoch) == len(datacsv):
            break
    j += 1

# -----FFT----- #
#Test avec sinus
#x = np.linspace(0.0, samples*exp_time, samples)
#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

fft_data = np.array(datacsv)
print("fftdata :", len(datacsv))
# ---Soustraction de la valeur neutre aux donnees--- #
fft_data = np.subtract(fft_data, ref_nose)

# ---Definition d'une epoch--- #
epoch = nb_epoch * [fft_data[0:ep_lgt, 1]]
print(np.shape(epoch))

# ---Definition d'un channel--- #
channel = nb_channels*[None]
for i in range(nb_channels):
    channel[i] = epoch[:]

print("longueur channel :", np.shape(channel))

# ---Integration des donnees dans array 3D [channel][epoch][echantillon]--- #
for chan in range(nb_channels):
    for epo in range(1, nb_epoch):
        channel[chan][epo] = fft_data[((ep_lgt*epo)-ep_lgt):(ep_lgt*epo), chan]
        #print("channel : {} | epoc : {} | value : {}".format(chan, ep, channel[chan][ep][0]))
    print("ep : {}, chan : {},".format(epo, chan), channel[10][8][0])

#print(fft_data[((ep_lgt*ep)-ep_lgt):(ep_lgt*ep), chan])
#print("echantillon 46, 3e epoch, 1 channel", channel[0][3])

#fft_data = datacsv[8070:8710, 6]
#for i in range(0, len(fft_data)):
#    fft_data[i] = [x-ref_nose for x in fft_data[i][:, ]]
#print(channel[10][10])
fft_channel_1 = np.fft.fft(channel[3][2])
#print(channel[0][8])
fig, ax = pyplot.subplots()
xf = np.linspace(0.0, freq_ech/2, int(samples/2))
ax.plot(xf, 2/samples * abs(fft_channel_1[:samples//2]))
#ax.plot(np.linspace(0, len(fft_data), freq_ech*5), fft_data)
pyplot.show()

fft_channels = 2/samples * abs(fft_channel_1[:samples//2])
#max_freq = [i for i, x in enumerate(fft_channels[5:]) if x == max(fft_channels[5:])]
max_freq = np.argsort(fft_channels)
max_freq = max_freq[::-1]

# ---Filtrage passe-bande des donnees--- #
sc.signal.butter(5, btype='bandpass')

#for f in fft_FC5

# Ajout des colonnes manquantes au CSV
df['Time:128Hz'] = timestamp
df['Epoch'] = epoch
df['Event Id'] = 0
df['Event Date'] = 0
df['Event Duration'] = 0
df['Ref_Nose'] = 4179.179

# Reorganisation des colonnes dans l'ordre d'OpenVibe
df = df[OV_labels]

# Conversion de la liste vers CSV
df.to_csv('U001ai.csv', index=False, header=True)
