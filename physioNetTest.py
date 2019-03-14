import pyedflib as edf
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import mne
import os
import pickle
from imblearn.over_sampling import SMOTE, ADASYN


row_col_mapping = {
        '56789_': 6, 
        'ABCDEF': 1,
        'AGMSY5': 7, 
        'BHNTZ6': 8,
        'CIOU17': 9,
        'DJPV28': 10,
        'EKQW39': 11, 
        'FLRX4_': 12,
        'GHIJKL': 2,
        'MNOPQR': 3, 
        'STUVWX': 4,
        'YZ1234': 5,
        'FLRX4#': 0,
        '56789#': 0
    }

def smooth(x, window_len):
	"""
	Python implementation of matlab's smooth function
	"""

	if window_len < 3:
		return x

	# Window length must be odd
	if window_len%2 == 0:
		window_len += 1
	
	w = np.ones(window_len)
	y = np.convolve(w, x, mode='valid') / len(w)
	y = np.hstack((x[:window_len//2], y, x[len(x)-window_len//2:]))

	for i in range(0, window_len//2):
		y[i] = np.sum(y[0 : i+i]) / ((2*i) + 1)

	for i in range(len(x)-window_len//2, len(x)):
		y[i] = np.sum(y[i - (len(x) - i - 1) : i + (len(x) - i - 1)]) / ((2*(len(x) - i - 1)) + 1)

	return y

def extractPhysioNet(file):
    
    f = edf.EdfReader(file)
    n = f.signals_in_file
    sig_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))

    for i in range(n):
        sigbufs[i, :] = f.readSignal(i)

    ann = f.readAnnotations()
    disp_ind = np.array(ann[0], dtype=float)
    stim = np.array(ann[2])

    ch_types = ['eeg'] * 64
    ch_types = np.hstack((ch_types, ['ref_meg', 'ref_meg', 'eog', 'eog', 'eog', 'eog']))

    info = mne.create_info(sfreq=2048, ch_names=sig_labels, montage='standard_1020', ch_types=ch_types)
    raw = mne.io.RawArray(sigbufs, info)

    raw.filter(0.1, 30)
    raw.notch_filter(freqs=50, filter_length='auto', phase='zero')
    raw.resample(250, npad="auto")
    raw = raw.pick_channels(['Fz', 'Cz', 'P3', 'P4', 'Pz', 'Oz', 'PO7', 'PO8'])

    target = stim[0][4]
    X = []
    y = []

    for ind, i in enumerate(stim):
        data = raw.get_data(start=int(disp_ind[ind] * 250), stop=int((disp_ind[ind] + 0.8) * 250))    
        temp_x = []
        
        for d in data:
            r = smooth(d, 7)
            r = sig.decimate(r, q=12)
            temp_x = np.hstack((temp_x, r))
        
        X.append(temp_x)
        
        if target in i:
            y.append(1)
        else:
            y.append(0)

    y = np.array(y)
    X = np.array(X)

    return (X, y)

def extractPhysioNet2(file):
    global row_col_mapping

    f = edf.EdfReader(file)
    n = f.signals_in_file
    sig_labels = f.getSignalLabels()
    sigbufs = np.zeros((n, f.getNSamples()[0]))

    for i in range(n):
        sigbufs[i, :] = f.readSignal(i)

    ann = f.readAnnotations()
    disp_ind = np.array(ann[0], dtype=float)
    stim = np.array(ann[2])

    ch_types = ['eeg'] * 64
    ch_types = np.hstack((ch_types, ['ref_meg', 'ref_meg', 'eog', 'eog', 'eog', 'eog']))

    info = mne.create_info(sfreq=2048, ch_names=sig_labels, montage='standard_1020', ch_types=ch_types, verbose=False)
    raw = mne.io.RawArray(sigbufs, info, verbose=False)

    raw.filter(0.1, 20, verbose=False)
    raw.resample(250, npad="auto", verbose=False)
    raw = raw.pick_types(eeg=True, meg=False, verbose=False)

    target = stim[0][4]
    X = []
    y = []

    for ind, i in enumerate(stim[2:-1]):
        if i not in row_col_mapping:
            continue

        data = raw.get_data(start=int(round(disp_ind[ind] * 250, 2)), stop=int(round((disp_ind[ind] * 250), 2) + 160.0))
        X.append(data)
        
        if target in i:            
            y.append(1)
        else:
            y.append(0)
    
    X = np.array(X)
    y = np.array(y)    

    return (X, y)




data = {"Train": {"X": np.array([]), "y": np.array([])}, "Test": {"X": np.array([]), "y": np.array([])}}

folders = {"Train":['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11'], "Test":['s12']}

for fold_type in folders:
    for f in folders[fold_type]:
        for file in os.listdir("/home/upamanyu/Documents/NTU_Creton/Data/physionet_erpbci/" + f):
            if not file.endswith(".edf"):
                continue
            print("\n\n")
            print("/home/upamanyu/Documents/NTU_Creton/Data/physionet_erpbci/" + f + "/" + file)

            temp_x, temp_y = extractPhysioNet2("/home/upamanyu/Documents/NTU_Creton/Data/physionet_erpbci/" + f + "/" + file)
            if data[fold_type]["X"].shape[0] == 0:
                data[fold_type]["X"] = temp_x
            else:
                data[fold_type]["X"] = np.vstack((data[fold_type]["X"], temp_x))    
            
            data[fold_type]["y"] = np.hstack((data[fold_type]["y"], temp_y))
            
            print("X:", np.array(data[fold_type]["X"]).shape)
            print("y:", data[fold_type]["y"].shape)


        with open("physionetCNNData.pickle", "ab") as m:
            print("Dumped Subject : ", f)
            pickle.dump(data, m)

        del(data)
        data = {"Train": {"X": np.array([]), "y": np.array([])}, "Test": {"X": np.array([]), "y": np.array([])}}

