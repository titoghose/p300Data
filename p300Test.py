import os
import mne
import pickle
import numpy as np 
import matplotlib.pyplot as plt    
import scipy.io as sio  
import scipy.signal as sig
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import normalize


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

def extract_data(data_file):
    
    if os.path.isfile(data_file):
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        return data

    sampling_rate = 250
    sampling_interval = sampling_rate / 1000.0

    data = sio.loadmat("/home/upamanyu/Documents/NTU_Creton/Data/kaggle_p300_Datase/P300S01.mat")["data"]

    eeg = np.transpose(data["X"][0][0])
    trial_points = np.squeeze(data["trial"][0][0])
    flash_points = data["flash"][0][0]

    trial_points = np.hstack((trial_points, eeg.shape[1]))

    cnt = 0
    temp_X = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[]}
    temp_y = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[]}
    
    for trial in trial_points[1:]:
        if cnt >= len(flash_points):
            break

        flash = flash_points[cnt][0]
        while flash <= trial:
            
            stimulation = flash_points[cnt][2]
            
            if len(temp_y[stimulation]) == 0:
                temp_y[stimulation] = [flash_points[cnt][3] - 1]
            else:
                temp_y[stimulation].append(flash_points[cnt][3] - 1)
            
            if len(temp_X[stimulation]) == 0:
                temp_X[stimulation] = np.array(eeg[:, flash : flash + int(sampling_interval*800)])
            else:    
                temp_X[stimulation] = np.dstack((temp_X[stimulation], (eeg[:, flash : flash + int(sampling_interval*800)])))
            
            cnt += 1
            if cnt >= len(flash_points):
                break
            flash = flash_points[cnt][0]

    data_X = []
    data_y = []

    for i in temp_X:
        temp_X[i] = np.swapaxes(temp_X[i], 1, 2)
        temp_X[i] = np.swapaxes(temp_X[i], 0, 1)

        for ind, j in enumerate(temp_X[i]):
            data = []
            for ch in range(8):
                smooth_data = smooth(j[ch], 7)
                dec = sig.decimate(smooth_data, q=12)
                data = np.hstack((data, dec))

            data_X.append(data)
            data_y.append(temp_y[i][ind])

    data_X = np.array(data_X)
    data_y = np.array(data_y)

    print(data_X.shape)
    print(data_y.shape)

    data = {"X":data_X, "y":data_y}

    with open("kaggle_p300.pickle", "wb") as f:
        pickle.dump(data, f)
    
    return data
    
# data_file = 'kaggle_p300.pickle'
data_file = "p300Data.p"
data = extract_data(data_file)

kfold = KFold(n_splits=10)

data["X"] = normalize(data["X"])
ind = np.where(data["y"] == 1)[0]

fig = plt.figure()
ax1 = fig.add_subplot(8, 1, 1)
ax2 = fig.add_subplot(8, 1, 2)
ax3 = fig.add_subplot(8, 1, 3)
ax4 = fig.add_subplot(8, 1, 4)
ax5 = fig.add_subplot(8, 1, 5)
ax6 = fig.add_subplot(8, 1, 6)
ax7 = fig.add_subplot(8, 1, 7)
ax8 = fig.add_subplot(8, 1, 8)

for i in range(0, len(ind), 8):
    ax1.plot(data["X"][i])
    ax2.plot(data["X"][i+1])
    ax3.plot(data["X"][i+2])
    ax4.plot(data["X"][i+3])
    ax5.plot(data["X"][i+4])
    ax6.plot(data["X"][i+5])
    ax7.plot(data["X"][i+6])
    ax8.plot(data["X"][i+7])
    plt.show()


# X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"], test_size=0.15, shuffle=True)

# lda_clf = LinearDiscriminantAnalysis(solver='lsqr')
# lsvm_clf = LinearSVC(class_weight='balanced', dual=False)
# mlp_clf = MLPClassifier(hidden_layer_sizes=(256, 128), verbose=True)

# lsvm_clf.fit(X_train, y_train)
# lda_clf.fit(X_train, y_train)
# mlp_clf.fit(X_train, y_train)

# pred_lsvm = lsvm_clf.predict(X_test)
# pred_lda = lda_clf.predict(X_test)
# pred_mlp = mlp_clf.predict(X_test)

# print("      F1 \t Prec \t Rec")
# print("LSVM: %0.3f \t %0.3f \t %0.3f" % (f1_score(y_test, pred_lsvm), precision_score(y_test, pred_lsvm), recall_score(y_test, pred_lsvm)))
# print("LDA : %0.3f \t %0.3f \t %0.3f" % (f1_score(y_test, pred_lda), precision_score(y_test, pred_lda), recall_score(y_test, pred_lda)))
# print("MLP : %0.3f \t %0.3f \t %0.3f" % (f1_score(y_test, pred_mlp), precision_score(y_test, pred_mlp), recall_score(y_test, pred_mlp)))

# for train_idx, test_idx in kfold.split(data["X"]):
#     X_train, y_train = data["X"][train_idx], data["y"][train_idx]
#     X_test, y_test = data["X"][test_idx], data["y"][test_idx]
    
#     svm_clf.fit(X_train, y_train)
#     lsvm_clf.fit(X_train, y_train)
#     lda_clf.fit(X_train, y_train)

#     print("%2.3f \t\t %2.3f \t\t %2.3f" % (lda_clf.score(X_test, y_test), svm_clf.score(X_test, y_test), lsvm_clf.score(X_test, y_test)))
