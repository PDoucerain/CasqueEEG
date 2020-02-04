import os
from scipy.io import loadmat
import Data_Analysis as dp

# ---Importation des donnees du .mat--- #
data_filename = 'U001ai.mat'
data_folder = 'SSVEP_Training/Dataset/'
data_path = data_folder + data_filename
data = loadmat(data_path)
success = 0
i = 0
verbose = False

for filename in os.listdir(data_folder):
    if filename[-5] != 'x':
        print(filename)
        verbose = False
        if filename == 'U005dii.mat':
            verbose = True
        if filename[-6:-4] == 'ii':
            seq = 1
        else:
            seq = 0
        data = loadmat(data_folder+filename)
        success += dp.dataAnalysis(data, seq, verbose)
        i += 1

print("Sucess rate : {0:.2f}%".format(success*100/i))
