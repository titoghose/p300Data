import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import mne

data = sio.loadmat("/home/upamanyu/Documents/NTU_Creton/Data/SEED-IV/eeg_raw_data/1/1_20160518.mat")
trial_1 = data["cz_eeg1"]

# fz = data["cz_eeg1"][9]
# fcz = data["cz_eeg1"][18]
# cz = data["cz_eeg1"][27]
# cpz = data["cz_eeg1"][36]
# pz = data["cz_eeg1"][45]
# poz = data["cz_eeg1"][53]
# oz = data["cz_eeg1"][59]

channel_indices = [9, 18, 27, 36, 45, 53, 59]
channel_names = ["fz", "fcz", "cz", "cpz", "pz", "poz", "oz"]
channel_types = ["eeg", "eeg", "eeg", "eeg", "eeg", "eeg", "eeg"]

n_channels = len(channel_names)
sampling_rate = 200
montage = 'standard_1020'

info = mne.create_info(ch_names=channel_names, sfreq=sampling_rate, ch_types=channel_types, montage=montage)

raw = mne.io.RawArray(trial_1[channel_indices], info)
raw = mne.io.RawArray(raw.get_data(start=0, stop=800), info)
raw.notch_filter(np.arange(50, 100, 50), filter_length='auto', phase='zero')
raw.filter(None, 30., fir_design='firwin')
raw.filter(0.3, None, fir_design='firwin')

# raw.plot(n_channels=7, butterfly=True, show=True, scalings='auto')
# raw.plot_psd()
# plt.show()

ica = mne.preprocessing.ICA(n_components=7, method='fastica', random_state=5)
ica.fit(raw)
ica.plot_components(picks=range(7), inst=raw)