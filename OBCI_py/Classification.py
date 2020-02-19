import numpy as np
import scipy as sc
from scipy import signal
import matplotlib.pyplot as plt


class Patient:
    def __init__(self, filename, data, seq, rank): #, chan, ep, sequence, possible_freq
        self.eeg_data = np.transpose(data['eeg'])
        self.events = data['events']
        self.patient = filename
        self.bp_wn=[]

        

        self.labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Status']
        self.sequence_plan = [[4, 2, 3, 5, 1, 2, 5, 4, 2, 3, 1, 5], [4, 3, 2, 4, 1, 2, 5, 3, 4, 1, 3, 1, 3]]

        # ---Data infos--- #
        self.nb_channels = len(self.eeg_data[0, :])-1
        self.nb_epoch = 0

        # ---Echantillonage--- #
        self.freq_ech = 128
        self.exp_time = 5
        self.time_step = 1 / self.freq_ech
        self.samples = self.freq_ech * self.exp_time
        self.ep_lgt = self.exp_time * self.freq_ech
        self.ref_nose = 4179.179
        self.PI = 3.1416
        self.channel = self.nb_channels * [None]

        for name in range(self.nb_channels):
            self.channel[name] = Channel(self.labels[name], seq, self.eeg_data[:, name], self.events, self.sequence_plan, rank)

        #self.dataAnalysis(self.eeg_data, self.events)

    def dataAnalysis(self, data, events, seq, verbose, filename):

        labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Status']

        # ---Frequences utilisees par SSVEP--- #
        f_SSVEP = [12.00, 10.00, 8.57, 7.50, 6.66]
        sequence_plan = [[4, 2, 3, 5, 1, 2, 5, 4, 2, 3, 1, 5], [4, 3, 2, 4, 1, 2, 5, 3, 4, 1, 3, 1, 3]]
        sequence_calc = []

        # ---Find trials start and ends--- #
        # print(np.shape(events))
        start_idx = np.where(events == 32779)[0]
        end_idx = np.where(events == 32780)[0]

        trial_start = []
        trial_end = []
        for idx in start_idx:
            trial_start.append(events[idx, 2])

        for idx in end_idx:
            trial_end.append(events[idx, 2])

        # trial_start = np.where(32779, events[3])
        # print("start", trial_start)

        for i in range(len(sequence_plan[seq])):
            sequence_plan[seq][i] = f_SSVEP[sequence_plan[seq][i] - 1]

        bp_freq = [f_SSVEP[-1] - 0.2, f_SSVEP[0] + 1]
        self.bp_wn = [i * 2.0 / self.freq_ech for i in bp_freq]  # for butterworth filter

        data = list(data)
        data = np.transpose(data)

        self.nb_channels = len(data[0])
        self.nb_epoch = len(trial_start)
        self.ep_lgt = trial_end[0] - trial_start[0]

        # ---Epoch definition--- #
        epoch_label = []
        j = 0
        for ii in range(int(len(datacsv) / samples) + 1):
            for jj in range(exp_time * freq_ech):
                epoch_label.append(j)
                if len(epoch_label) == len(datacsv):
                    break
            j += 1

        fft_data = np.array(data)

        # ---Soustraction de la valeur neutre aux donnees--- #
        fft_data = self.referenceData(fft_data)

        # ---Definition d'une epoch--- #
        epoch = nb_epoch * [fft_data[0:ep_lgt, 1]]

        # ---Definition d'un channel--- #
        channel = nb_channels * [None]
        for i in range(nb_channels):
            channel[i] = epoch[:]

        # ---Integration des donnees dans array 3D [channel][epoch][echantillon]--- #
        for c in range(nb_channels):
            for e in range(1, nb_epoch):
                channel[c][e] = fft_data[trial_start[e]:trial_end[e], c]

        for epo in range(nb_epoch):
            # ---Choix du channel et de l'epoch--- #
            chan = 6  # channel 6 et 7 pour lobe occipital
            freq_calc = self.dataFilter(channel[chan][epo])
            sequence_calc.append(freq_calc)
            Classification.statsReport(fft_channels_analysis, chan, epo, sequence_plan[seq], f_SSVEP)

            # print("Fréquence prédominente : {} Hz".format(np.argmax(fft_channels_analysis) / samples * freq_ech))

        sucess = abs(np.subtract(sequence_plan[seq], sequence_calc[:]))

        sucess_rate = 0
        for s in sucess:
            if s < 0.6:
                sucess_rate += 1
        sucess_rate /= nb_epoch

        return sucess_rate

    def referenceData(self, data):
        ref = data.mean()
        data = np.subtract(data, ref)

        return data

    def openVibeFormat(self, data):
        labels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Status']
        OV_labels = ['Time:128Hz', 'Epoch', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4',
                     'F8',
                     'AF4', 'Status', 'Ref_Nose', 'Event Id', 'Event Date', 'Event Duration']

        # ---Ajout des labels des données--- #
        df = pd.DataFrame(data=data, columns=labels)
        timestamp = np.linspace(0, len(data) / self.freq_ech, len(data))

        # Ajout des colonnes manquantes au CSV
        df['Time:128Hz'] = timestamp
        df["Epoch"] = epoch_label
        df['Event Id'] = 0
        df['Event Date'] = 0
        df['Event Duration'] = 0
        df['Ref_Nose'] = 4179.179

        # Reorganisation des colonnes dans l'ordre d'OpenVibe
        df = df[OV_labels]

        # Conversion de la liste vers CSV
        df.to_csv(data_filename, index=False, header=True)

    def addNewData(self, data, seq):
        eeg_data = np.transpose(data['eeg'])
        events = data['events']
        for name in range(self.nb_channels):
            self.channel[name].addNewData(eeg_data[:, name], events, seq)

    def dataFilter(self, data):
        # ---Filtrage passe-bande des donnees entre 6.66 et 12 Hz--- #
        [b, a] = sc.signal.butter(7, bp_wn, btype='bandpass')
        fft_channel_1 = signal.filtfilt(b, a, data)
        fft_channel_1 = np.fft.fft(fft_channel_1)
        fft_channels_analysis = abs(fft_channel_1[:samples // 2])
        freq_predom = np.argmax(fft_channels_analysis) / samples * freq_ech
        freq_arr = abs(np.subtract(f_SSVEP, freq_predom))
        freq_idx = int(np.argmin(freq_arr))
        freq_calc = f_SSVEP[freq_idx]

    def displayData(self, type):
        # if sucess_rate < 0.05:
        #     print(filename)
        #     print("Taux de réussite : {0:.2f} %".format(sucess_rate * 100))
        #
        # if sucess_rate < 0.05:
        #     print("nb_channels :", nb_channels)
        #     print("nb_epoch :", nb_epoch)
        #
        #     print("Sequence calculée :", sequence_calc)
        #     print("Sequence planifiée :", sequence_plan[seq])
        #
        #     print("Taux de réussite : {0:.2f} %".format(sucess_rate * 100))
        #     chan = 6  # channel 6 et 7 pour lobe occipital
        #     epo = 3
        #     # ---fft plot--- #
        #     fig, ax = plt.subplots()
        #     xf = np.linspace(0.0, freq_ech / 2, int(samples / 2))
        #     ax.plot(xf, 2 / samples * abs(fft_channel_1[:samples // 2]))
        #     ax.set_ylabel('uvolts')
        #     ax.set_xlabel('Hz')
        #     ax.set_title('Channel eeg {}, epoch : {}'.format(labels[chan], epo + 1))
        #     # ax.plot(np.linspace(0, len(fft_data), samples/2), fft_channels_analysis)
        #     plt.show()

        # fig, ax = plt.subplots()
        # xf = np.linspace(0.0, freq_ech/2, int(samples/2))
        # ax.plot(xf, 2/samples * abs(fft_channel_1[:samples//2]))
        # ax.set_ylabel('uvolts')
        # ax.set_xlabel('Hz')
        # ax.set_title('Channel eeg {}, epoch : {}'.format(labels[chan], epo+1))
        # ax.plot(np.linspace(0, len(fft_data), freq_ech*5), fft_data)
        # plt.show()
        data = type
        # ---Spectrum Analysis--- #
        [b, a] = sc.signal.butter(5, self.bp_wn, btype='bandpass')
        fft_channel_1 = signal.filtfilt(b, a, data)
        f, t, Zxx = sc.signal.spectrogram(fft_channel_1, freq_ech, window=('tukey', 0.25), nperseg=1024)
        plt.pcolormesh(t, f, Zxx)
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

    def successRate(self, rank):
        seq_calc = self.channel[6].getSequence(rank)
        suc = 0
        for s in seq_calc:
            if s:
                suc += 1
            else:
                suc += 0
        suc /= len(seq_calc)


        return suc*100

    def successRateImprovements(self):
        # ---ESSAI : Plusieurs bp filter pour les 5 freq--- #
        # fft_channel_multi = len(f_SSVEP) * [0]
        # for ii in range(len(f_SSVEP)):
        #     bp_freq = [f_SSVEP[-ii]-0.2, f_SSVEP[-ii]+0.2]
        #     bp_wn = [i * 2.0 / freq_ech for i in bp_freq]  # for butterworth filter
        #     [b, a] = sc.signal.butter(7, bp_wn, btype='bandpass')
        #     fft_channel_multi[ii] = signal.filtfilt(b, a, channel[chan][epo])
        #
        # fft_channel_1 = [0]
        # ii = 0
        # for i in range(len(f_SSVEP)//2):
        #     fft_channel_1 += fft_channel_multi[ii] + fft_channel_multi[ii+1]
        #     ii += 2
        # fft_channel_1 = fft_channel_multi[3] + fft_channel_multi[0] + fft_channel_multi[1] + fft_channel_multi[2] + fft_channel_multi[4]
        # fft_channel_1 = np.fft.fft(fft_channel_1)
        # fft_channels_analysis = abs(fft_channel_1[:samples // 2])
        # freq_predom = np.argmax(fft_channels_analysis) / samples * freq_ech
        # freq_arr = abs(np.subtract(f_SSVEP, freq_predom))
        # freq_idx = int(np.argmin(freq_arr))
        # freq_calc = f_SSVEP[freq_idx]

        # ---ESSAI : Soustraction des fréquences des canaux environnants--- #
        # fft_channel_2 = signal.filtfilt(b, a, channel[chan - 1][epo])
        # fft_channel_2 = np.fft.fft(fft_channel_2)
        # fft_channel_1 = np.subtract(fft_channel_1, fft_channel_2)

        # ---ESSAI : Selection du canal qui a la plus grande amplitude (certitude des données) entre O1 et O1--- #
        # freq_max_chans = -8000
        # for chan in range(6, 8):
        #     # ---Filtrage passe-bande des donnees--- #
        #     [b, a] = sc.signal.butter(6, bp_wn, btype='bandpass')
        #     fft_channel_1 = signal.filtfilt(b, a, channel[chan][epo])
        #     fft_channel_1 = np.fft.fft(fft_channel_1)
        #
        #     # ---Detection de la frequence predominante--- #
        #     freq_max_channel = fft_channels_analysis[np.argmax(fft_channels_analysis)]
        #
        #     if freq_max_channel > freq_max_chans:
        #         freq_max_chans = freq_max_channel
        #         freq_predom = np.argmax(fft_channels_analysis) / samples * freq_ech    #modifier sequence_calc.append(freq_predom)

        # ---ESSAI : Delimitation uniquement aux frequences utilisees--- #
        # freq_admissible = np.multiply(f_SSVEP, samples/freq_ech)
        #
        # frequence_dominente = -8000                                             # initie a une valeur tres petite
        # for f in freq_admissible:
        #     frequences_actuelle = fft_channels_analysis[int(f)-2:int(f)+2]      # +- autour de la valeur admissible
        #     max_frequence = max(frequences_actuelle)                            # amplitude max des valeurs autour
        #     if max_frequence > frequence_dominente:
        #         frequence_dominente = max_frequence
        #         frequence_calcule = f/5                                       # frequence deduite (entre 6.66 et 12)
        #
        # freq_arr = abs(np.subtract(f_SSVEP, freq_predom))
        # freq_idx = int(np.argmin(freq_arr))
        # freq_calc = f_SSVEP[freq_idx]                                    # Mettre sequence_calc.append(frequence_calc)

        # ---ESSAI : Evaluer le succes selon les extremes des frequences--- #
        # # commenter freq_arr, freq_idx et freq_calc environ ligne 100
        # # remplacer sequence_calc.append(freq_calc) par sequence_calc.append(freq_predom)
        # sucess_rate = 0
        #
        # if seq == 0:
        #     sequence_min = [sequence_calc[3], sequence_calc[6], sequence_calc[11]]
        #     sequence_max = [sequence_calc[4], sequence_calc[10]]
        # if seq == 1:
        #     sequence_min = [sequence_calc[6]]
        #     sequence_max = [sequence_calc[4], sequence_calc[9], sequence_calc[11]]
        #
        # for item in sequence_min:
        #     if item < 9.33:
        #         sucess_rate += 1
        #
        # for item in sequence_max:
        #     if item >= 9.33:
        #         sucess_rate += 1
        # sucess_rate /= (len(sequence_min)+len(sequence_max))
        pass

class Channel:
    def __init__(self, name, seq, data, events, sequence_plan, rank):
        self.channel_name = name
        self.epoch = [[], []]                   #afin d'avoir 12 epoch ds seq 1 et 13 epoch ds seq 2
        self.epoch_start = [[], []]
        self.epoch_end = [[], []]
        self.rank = rank
        self.sequence_plan = sequence_plan
        self.addNewData(data, events, seq)

    def addNewData(self, data, events, seq):
        self.epochSeparation(events, seq)
        if self.channel_name == 'O23':
            filtered_data = self.adaptativeFilter(64, data)
        else:
            filtered_data = data
        for i in range(len(self.epoch_start[seq])):
            self.epoch[seq].append(Epoch(filtered_data[self.epoch_start[seq][i]:self.epoch_end[seq][i]], self.sequence_plan[seq][i], self.rank))

    def epochSeparation(self, data, seq):
        start_idx = np.where(data == 32779)[0]
        end_idx = np.where(data == 32780)[0]

        i = 0
        for idx in start_idx:
            self.epoch_start[seq].append(data[idx][2])
            i += 1

        i = 0
        for idx in end_idx:
            self.epoch_end[seq].append(data[idx][2])
            i += 1

    def getSequence(self, rank):
        sequence_calc = []
        for s in self.epoch:
            for ep in s:
                sequence_calc.append(ep.getSuccess(rank))

        return sequence_calc

    def adaptativeFilter(self, filter_order, raw_data):
        M = filter_order  # longueur de la fenetre glissante (longeur du X) fs/M pour avoir la plus basse frequence filtrable
        u = raw_data
        step = 0.07
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

        x_total = np.append(raw_data[:M], x_total)
        x_total = np.append(x_total, raw_data[-1])
        return np.array(x_total)

class Epoch:
    def __init__(self, data, ref_value, rank):
        self.top_X_frequency = []
        self.freq_ech = 128
        self.sample_nb = 640
        self.frequencies = [12.00, 10.00, 8.57, 7.50, 6.66]
        self.bp_freq = [self.frequencies[-1]-0.2, self.frequencies[0]+1]
        self.bp_wn = [i * 2.0 / self.freq_ech for i in self.bp_freq]
        self.success = None
        self.ref_value = ref_value
        self.raw_data = data
        self.fft_data = []

        self.dataProcessing(data, rank)

    def dataProcessing(self, data, rank):
        filtered_data = self.dataFilter(data)
        self.fft_data = self.fft(filtered_data)
        self.statsReport(self.fft_data, rank)

    def dataFilter(self, data):
        # ---Filtrage passe-bande des donnees entre 6.66 et 12 Hz--- #
        [b, a] = sc.signal.butter(7, self.bp_wn, btype='bandpass')
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data

    def fft(self, data):
        fft_data = np.fft.fft(data)
        fft_data = abs(fft_data[:self.sample_nb // 2])
        return fft_data

    def statsReport(self, data, rank):
        max_frequency = self.findMaximumValues(data, rank)
        max_frequency = np.multiply(max_frequency, 128/640)

        self.top_X_frequency = self.closestChoice(self.frequencies, max_frequency)

        # best_value_diff = abs(np.subtract(max_frequency, self.ref_value))
        # best_value_idx = np.argmin(best_value_diff)
        # best_value = max_frequency[best_value_idx]

    def findMaximumValues(self, values, rank):
        arr = np.argsort(values)

        maximum_values = arr[-rank:]

        return maximum_values

    def closestChoice(self, choices, values):
        top_freqs = []
        for v in values:
            diff = abs(np.subtract(choices, v))
            top_freqs.append(choices[np.argmin(diff)])

        return top_freqs #descending order (worst to best)

    def getSuccess(self, pos):
        for value in self.top_X_frequency[-pos:]:
            if value == self.frequencies[self.ref_value-1]:
                self.success = True
                break
            else:
                self.success = False
        return self.success

    def displayData(self, data):
        # ---Spectrum Analysis--- #
        [b, a] = sc.signal.butter(5, self.bp_wn, btype='bandpass')
        fft_channel_1 = signal.filtfilt(b, a, data)
        f, t, Zxx = sc.signal.spectrogram(fft_channel_1, self.freq_ech, window=('tukey', 0.25), nperseg=1024)
        plt.pcolormesh(t, f, Zxx)
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

