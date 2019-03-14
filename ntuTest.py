import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


guilty_subjects = ["/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Guilty Subject 06.csv",
"/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Guilty Subject 07.csv",
"/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Guilty Subject 08.csv",
"/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Guilty Subject 09.csv",
"/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Guilty Subject 10.csv",
"/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Guilty Subject 11.csv",]

innocent_subjects = ["/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Innocent Subject 06.csv",
"/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Innocent Subject 07.csv",
"/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Innocent Subject 08.csv",
"/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Innocent Subject 09.csv",
"/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Innocent Subject 10.csv",
"/home/upamanyu/Documents/NTU_Creton/Data/csv_files_with_column_headers/Innocent Subject 11.csv"]

def evoked_array(subjects, stimuli):
    all_data = []

    for sub in subjects:
        df = pd.read_csv(sub, usecols=["StimulusName", "EventSource", "O1/Pz (Epoc)", "O2 (Epoc)"])
        eeg = []

        for s in stimuli:
            
            stim_rows = np.where(df.StimulusName.str.contains(s) & df.EventSource.str.contains("Raw EEG"))[0]
            if len(stim_rows) < 200 :
                continue

            data_o1 = np.array(df["O1/Pz (Epoc)"][stim_rows], dtype=float)
            data_o2 = np.array(df["O2 (Epoc)"][stim_rows], dtype=float)

            data_o1 = data_o1[:200]
            data_o2 = data_o2[:200]

            mean_o1 = np.mean(data_o1)
            mean_o2 = np.mean(data_o2)

            data_o1 -= mean_o1
            data_o2 -= mean_o2

            data = np.average(np.vstack((data_o1, data_o2)), axis=0)
            data = np.expand_dims(data, axis=0)

            channel_names = ["Oz"]
            channel_types = ["eeg"]
            sfreq = 256

            info = mne.create_info(ch_names=channel_names, ch_types=channel_types, sfreq=sfreq, montage='standard_1020')
            raw = mne.io.RawArray(data, info)
            raw.notch_filter(np.arange(60, 128, 60), filter_length='auto', phase='zero')
            raw.filter(0.1, 30., fir_design='firwin')

            if len(eeg) == 0:
                eeg = np.array(raw.get_data(start=0, stop=200))
            else:
                eeg = np.vstack((eeg, raw.get_data(start=0, stop=200)))
        
        eeg = np.average(eeg, axis=0)
        all_data.append(eeg)

    return all_data

rel_stimuli = ["Jon_Secretary2-1-1","Jon_Secretary2-1-2","Jon_Secretary2-1-3","Jon_Secretary2-1-4","Jon_Secretary2-1-5","Jon_Secretary2-1-6","Jon_Secretary2-1-7","Jon_Secretary2-1-8","Jon_Secretary2-1-9","Jon_Secretary2-1-10","Jon_Secretary2-1-11","Wallet-1-1","Wallet-1-2","Wallet-1-3","Wallet-1-4","Wallet-1-5","Wallet-1-6","Wallet-1-7","Wallet-1-8","Wallet-1-9","Wallet-1-10","Wallet-1-11"]

irr_stimuli= ["Irrelevant_Room-1-1","Irrelevant_Room-1-2","Irrelevant_Room-1-3","Irrelevant_Room-1-4","Irrelevant_Room-1-5","Irrelevant_Room-1-6","Irrelevant_Room-1-7","Irrelevant_Room-1-8","Irrelevant_Room-1-9","Irrelevant_Room-1-10","Irrelevant_Room-1-11","Irrelevant_Secretary2-1-1","Irrelevant_Secretary2-1-2","Irrelevant_Secretary2-1-3","Irrelevant_Secretary2-1-4","Irrelevant_Secretary2-1-5","Irrelevant_Secretary2-1-6","Irrelevant_Secretary2-1-7","Irrelevant_Secretary2-1-8","Irrelevant_Secretary2-1-9","Irrelevant_Secretary2-1-10","Irrelevant_Secretary2-1-11"]

rel_inn = evoked_array(innocent_subjects, rel_stimuli)
rel_guil = evoked_array(guilty_subjects, rel_stimuli)

irrel_inn = evoked_array(innocent_subjects, irr_stimuli)
irrel_guil = evoked_array(guilty_subjects, irr_stimuli)

rel_data = np.vstack((rel_inn, rel_guil))
irrel_data = np.vstack((irrel_inn, irrel_guil))

fig = plt.figure()
n = len(rel_inn) + len(rel_guil)
ax = [None] * n

for i in range(n):
    ax[i] = fig.add_subplot(n, 1, i+1)
    x = np.arange(0, 200) * 4
    y = rel_data[i]
    y2 = irrel_data[i]
    ax[i].plot(x, y)
    ax[i].plot(x, y2, linestyle='--')
    plt.axvline(x=300, c='r', linestyle='--')
    plt.axvline(x=600, c='r', linestyle='--')

plt.show()