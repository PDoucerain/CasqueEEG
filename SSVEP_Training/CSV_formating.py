import numpy as np
import pandas as pd
import scipy.io as sc
from scipy.io import loadmat
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle
import csv

#data = pd.read_csv("EEG_CSV/01-signal.csv")
#x = loadmat("EEG_CSV/U001ai")

#sorted(x.keys())
#csv.writer(data.csv)
labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Status']
OV_labels = ['Time:128Hz', 'Epoch', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Status', 'Ref_Nose', 'Event Id', 'Event Date', 'Event Duration']
data = loadmat("Dataset/U001aii.mat")
freq_ech = 128
exp_time = 5
timestep = 1/freq_ech
data = data['eeg']
#samples = freq_ech*exp_time
samples = 640


datacsv = list(data)
datacsv = np.transpose(datacsv)


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

#-----FFT-----#
#Test avec sinus
#x = np.linspace(0.0, samples*exp_time, samples)
#y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)

#fft_data = datacsv[0:freq_ech*5, 8]
fft_data = datacsv[8070:8710, 9]
fft_data = [x-4179.179 for x in fft_data]
fft_FC5 = np.fft.fft(fft_data)
fig, ax = pyplot.subplots()
xf = np.linspace(0.0, freq_ech/2, int(samples/2))
ax.plot(xf, 2/samples * abs(fft_FC5[:samples//2]))
#ax.plot(np.linspace(0, len(fft_data), freq_ech*5), fft_data)
pyplot.show()

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
