import os
from scipy.io import loadmat
import Classification as cl
import numpy as np

# ---Importation des donnees du .mat--- #
data_filename = 'U001ai.mat'
data_folder = 'SSVEP_Training/Dataset/'
data_path = data_folder + data_filename
data = loadmat(data_path)
success = 0
i = 0
verbose = False
registered_patients = []
patients = []
suc_rate = []
rank = 5
average_success = []
for i in range(rank):
    average_success.append([0])

for filename in os.listdir(data_folder):
    if filename[-5] != 'x':
        verbose = False
        if filename == 'U005aii.mat':
            verbose = True
        if filename[-6:-4] == 'ii':
            seq = 1
        else:
            seq = 0
        data = loadmat(data_folder+filename)
        i += 1
        if filename[:5] in registered_patients:
            patient_idx = registered_patients.index(filename[:5])
            patients[patient_idx].addNewData(data, seq)
            for i in range(rank):
                suc_rate = patients[registered_patients.index(filename[:5])].successRate(i+1)
                average_success[i] = np.add(average_success[i], suc_rate)
            #print("Patient {}, success rate : {} %".format(filename, suc_rate))

        else:
            registered_patients.append(filename[:5])
            patients.append(cl.Patient(filename[:5], data, seq, rank))

for i in range(rank):
    n = average_success[i][0]
    n /= len(registered_patients)
    print("Sucess rate rank {} : {}%".format(i+1, n))

