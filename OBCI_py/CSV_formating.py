import os
from scipy.io import loadmat
import Classification as cl
import numpy as np
import GUI


class Report:
    def __init__(self):
        self.names = []
        self.settings = []
        self.settings_val = []

        # ---Importation des donnees du .mat--- #
        data_filename = 'U001ai.mat'
        data_folder = 'SSVEP_Training/Dataset/'
        data_path = data_folder + data_filename
        data = loadmat(data_path)
        success = 0
        i = 0
        verbose = False
        self.registered_patients = []
        self.patients = []
        patients = []
        suc_rate = []
        self.success_by_freq = [0, 0, 0, 0, 0]

        rank = 1
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
                data = loadmat(data_folder + filename)
                i += 1
                if filename[:5] in self.registered_patients:
                    patient_idx = self.registered_patients.index(filename[:5])
                    self.patients[patient_idx].addNewData(data, seq)
                    for i in range(rank):
                        suc_rate = self.patients[self.registered_patients.index(filename[:5])].successRate(i + 1)
                        average_success[i] = np.add(average_success[i], suc_rate)
                        # print("Patient {}, success rate : {} %".format(filename, suc_rate))

                else:
                    self.registered_patients.append(filename[:5])
                    self.patients.append(cl.Patient(filename[:5], data, seq, rank))

        for i in range(rank):
            n = average_success[i][0]
            n /= len(self.registered_patients)
            print("Sucess rate rank {} : {}%".format(i + 1, n))

        self.successRate()

        self.frame = GUI.MainApplication(self.registered_patients, self.patients[0].getChannelsName(), self.success_by_freq)
        self.window = self.frame.frames[GUI.MainWindow]
        LW = self.frame.frames[GUI.MainWindow].lengthw
        mwindow_size = str(1260 + LW) + 'x' + str(350)
        self.frame.geometry(mwindow_size)
        self.frame.mainloop()
        self.guiLoop()

    def successRate(self):
        for patient in self.patients:
            patient.setOverallSuccess()
            self.success_by_freq = np.add(self.success_by_freq, patient.overall_success)
        self.success_by_freq = np.divide(self.success_by_freq, len(self.patients))

    def generateReport(self):
        # Afficher fft sur une epoch
        if 'selected' in self.settings[0]:
            self.patients[self.registered_patients.index(self.names[0])].displayFFT(self.settings_val[0], self.settings_val[1])

        # Afficher spectrum
        if 'selected' in self.settings[1]:
            print('spectrum')

        # Afficher Taux de succes
        if 'selected' in self.settings[2]:
            print('success rate')

        # Afficher les stats frequences
        if 'selected' in self.settings[3]:
            print('frequency report')

        else:
            pass

    def getSucessByFrequency(self):
        success_by_freq = []

    def guiLoop(self):
        request = self.window.request
        while True:
            if self.window.ready_to_send:
                self.names = request[0]
                self.settings = request[1]
                self.settings_val = request[2]
                self.generateReport()
                self.window.ready_to_send = False
                self.frame.mainloop()
            else:
                break


report_request = Report()
report_request.generateReport()

# # ---Importation des donnees du .mat--- #
# data_filename = 'U001ai.mat'
# data_folder = 'SSVEP_Training/Dataset/'
# data_path = data_folder + data_filename
# data = loadmat(data_path)
# success = 0
# i = 0
# verbose = False
# registered_patients = []
# patients = []
# suc_rate = []
# rank = 1
# average_success = []
# for i in range(rank):
#     average_success.append([0])
#
# for filename in os.listdir(data_folder):
#     if filename[-5] != 'x':
#         verbose = False
#         if filename == 'U005aii.mat':
#             verbose = True
#         if filename[-6:-4] == 'ii':
#             seq = 1
#         else:
#             seq = 0
#         data = loadmat(data_folder+filename)
#         i += 1
#         if filename[:5] in registered_patients:
#             patient_idx = registered_patients.index(filename[:5])
#             patients[patient_idx].addNewData(data, seq)
#             for i in range(rank):
#                 suc_rate = patients[registered_patients.index(filename[:5])].successRate(i+1)
#                 average_success[i] = np.add(average_success[i], suc_rate)
#                 #print("Patient {}, success rate : {} %".format(filename, suc_rate))
#
#         else:
#             registered_patients.append(filename[:5])
#             patients.append(cl.Patient(filename[:5], data, seq, rank))
#
# for i in range(rank):
#     n = average_success[i][0]
#     n /= len(registered_patients)
#     print("Sucess rate rank {} : {}%".format(i+1, n))

# frame = GUI.MainApplication(registered_patients, patients[0].getChannelsName())
# window = frame.frames[GUI.MainWindow]
# LW = frame.frames[GUI.MainWindow].lengthw
# mwindow_size = str(1260+LW) + 'x' + str(180 + (GUI.MainWindow.nb_pieces // 8) * 25)
# frame.geometry(mwindow_size)
# frame.mainloop()
# while True:
#     if window.ready_to_send:
#         report_request = Report(window.request)
#         report_request.generateReport()
#         window.ready_to_send = False
#         frame.mainloop()
#     else:
#         break

