import numpy as np
import pandas as pd
import scipy as sc
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt

def dataAnalysis(data, seq, verbose):
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

    labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Status']
    OV_labels = ['Time:128Hz', 'Epoch', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8',
                 'AF4', 'Status', 'Ref_Nose', 'Event Id', 'Event Date', 'Event Duration']

    # ---Frequences utilisees par SSVEP--- #
    f_SSVEP = [12.00, 10.00, 8.57, 7.50, 6.66]
    sequence_plan = [[4, 2, 3, 5, 1, 2, 5, 4, 2, 3, 1, 5], [4, 3, 2, 4, 1, 2, 5, 3, 4, 1, 3, 1, 3]]
    sequence_calc = []

    # ---Find trials start and ends--- #
    #print(np.shape(events))
    start_idx = np.where(events == 32779)[0]
    end_idx = np.where(events == 32780)[0]

    trial_start = []
    trial_end = []
    for idx in start_idx:
        trial_start.append(events[idx, 2])

    for idx in end_idx:
        trial_end.append(events[idx, 2])

    #trial_start = np.where(32779, events[3])
    #print("start", trial_start)

    for i in range(len(sequence_plan[seq])):
        sequence_plan[seq][i] = f_SSVEP[sequence_plan[seq][i]-1]

    bp_freq = [f_SSVEP[-1], f_SSVEP[0]]
    bp_wn = [i * 2.0/freq_ech for i in bp_freq]         # for butterworth filter

    datacsv = list(datacsv)
    datacsv = np.transpose(datacsv)

    nb_channels = len(datacsv[0])
    nb_epoch = len(trial_start)

    # ---Ajout des labels des données--- #
    df = pd.DataFrame(data=datacsv, columns=labels)
    timestamp = np.linspace(0, len(datacsv)/freq_ech, len(datacsv))

    # ---Epoch definition--- #
    epoch_label = []
    j = 0
    for ii in range(int(len(datacsv)/samples)+1):
        for jj in range(exp_time*freq_ech):
            epoch_label.append(j)
            if len(epoch_label) == len(datacsv):
                break
        j += 1

    fft_data = np.array(datacsv)
    #print("fftdata :", len(datacsv))
    # ---Soustraction de la valeur neutre aux donnees--- #
    fft_data = np.subtract(fft_data, ref_nose)

    # ---Definition d'une epoch--- #
    epoch = nb_epoch * [fft_data[0:ep_lgt, 1]]

    # ---Definition d'un channel--- #
    channel = nb_channels*[None]
    for i in range(nb_channels):
        channel[i] = epoch[:]

    #print("longueur channel :", np.shape(channel))

    # ---Integration des donnees dans array 3D [channel][epoch][echantillon]--- #
    for c in range(nb_channels):
        for e in range(1, nb_epoch):
            channel[c][e] = fft_data[trial_start[e]:trial_end[e], c]


    for epo in range(nb_epoch):
        # ---Choix du channel et de l'epoch--- #
        chan = 7    #channel 6 et 7 pour lobe occipital

        freq_max_chans = -8000
        for chan in range(6, 7):
            # ---Filtrage passe-bande des donnees--- #
            [b, a] = sc.signal.butter(6, bp_wn, btype='bandpass')
            fft_channel_1 = signal.filtfilt(b, a, channel[chan][epo])
            fft_channel_1 = np.fft.fft(fft_channel_1)

            fft_channel_2 = signal.filtfilt(b, a, channel[chan-1][epo])
            fft_channel_2 = np.fft.fft(fft_channel_2)
            #fft_channel_1 = np.subtract(fft_channel_1, fft_channel_2)
            #fft_channel_test = list(abs(fft_channel_1))

            #print(max(fft_channel_test))
            #fft_channel_test.index()

            # ---Detection de la frequence predominante--- #

            fft_channels_analysis = abs(fft_channel_1[:samples // 2])
            # ---Delimitation a seulement les frequences utilisees--- #
            freq_admisible = np.multiply(f_SSVEP, samples/freq_ech)

            frequence_dominente = -8000
            for f in freq_admisible:
                frequences_actuelle = fft_channels_analysis[int(f)-2:int(f)+2]
                max_frequence = max(frequences_actuelle)
                if max_frequence > frequence_dominente:
                    frequence_dominente = max_frequence
                    frequence_calcule = f/5
            freq_max_channel = fft_channels_analysis[np.argmax(fft_channels_analysis)]

            if freq_max_channel > freq_max_chans:
                freq_max_chans = freq_max_channel
                freq_predom = np.argmax(fft_channels_analysis) / samples * freq_ech

        freq_arr = abs(np.subtract(f_SSVEP, freq_predom))
        freq_idx = int(np.argmin(freq_arr))
        freq_calc = f_SSVEP[freq_idx]
        sequence_calc.append(frequence_calcule)

            #print("Fréquence prédominente : {} Hz".format(np.argmax(fft_channels_analysis) / samples * freq_ech))

    # ---Spectrum Analysis--- #
    [b, a] = sc.signal.butter(5, bp_wn, btype='bandpass')
    fft_channel_1 = signal.filtfilt(b, a, fft_data[:, 6])
    f, t, Zxx = sc.signal.spectrogram(fft_channel_1, freq_ech, window=('tukey', 0.25), nperseg=1024)
    # plt.pcolormesh(t, f, Zxx)
    # plt.title('STFT Magnitude')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    #plt.show()

    sucess = abs(np.subtract(sequence_plan[seq], sequence_calc[:]))
    sucess_rate = 0
    for s in sucess:
        if s < 0.6:
            sucess_rate += 1
    sucess_rate /= nb_epoch


    # fig, ax = plt.subplots()
    # xf = np.linspace(0.0, freq_ech/2, int(samples/2))
    # ax.plot(xf, 2/samples * abs(fft_channel_1[:samples//2]))
    # ax.set_ylabel('uvolts')
    # ax.set_xlabel('Hz')
    # ax.set_title('Channel eeg {}, epoch : {}'.format(labels[chan], epo+1))
    #ax.plot(np.linspace(0, len(fft_data), freq_ech*5), fft_data)
    #plt.show()

    # Ajout des colonnes manquantes au CSV
    # df['Time:128Hz'] = timestamp
    # df["Epoch"] = epoch_label
    # df['Event Id'] = 0
    # df['Event Date'] = 0
    # df['Event Duration'] = 0
    # df['Ref_Nose'] = 4179.179

    # Reorganisation des colonnes dans l'ordre d'OpenVibe
    #df = df[OV_labels]

    # Conversion de la liste vers CSV
    #df.to_csv(data_filename, index=False, header=True)
    print("Taux de réussite : {0:.2f} %".format(sucess_rate * 100))

    if verbose:
        print("nb_channels :", nb_channels)
        print("nb_epoch :", nb_epoch)

        print("Sequence calculée :", sequence_calc)
        print("Sequence planifiée :", sequence_plan[seq])

        print("Taux de réussite : {0:.2f} %".format(sucess_rate * 100))
        chan = 6  # channel 6 et 7 pour lobe occipital
        epo = 0
        # ---fft plot--- #
        # fig, ax = plt.subplots()
        # xf = np.linspace(0.0, freq_ech/2, int(samples/2))
        # ax.plot(xf, 2/samples * abs(fft_channel_1[:samples//2]))
        # ax.set_ylabel('uvolts')
        # ax.set_xlabel('Hz')
        # ax.set_title('Channel eeg {}, epoch : {}'.format(labels[chan], epo+1))
        # ax.plot(np.linspace(0, len(fft_data), samples/2), fft_channels_analysis)
        # plt.show()

    return sucess_rate