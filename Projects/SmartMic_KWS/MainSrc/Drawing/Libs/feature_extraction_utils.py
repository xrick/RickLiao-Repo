import numpy as np
from numpy.linalg import norm
import librosa

"""
global variables
"""
fs = 16000
windowSize = fs * 0.025
windowStep = fs * 0.010
nDims = 40
context_l = 30
context_r = 10

def vad_test(s, fs):
    s = s - np.amin(s)
    s = s / np.amax(s)
    FrameSize = int(fs * 0.025)  # 400
    ShiftSize = int(fs * 0.010)  # 160
    Overlap = FrameSize - ShiftSize  # 240
    threshold = -1.9
    s_temp = []
    temp = []
    temp_all = []
    new = []
    rest_s = []
    t = s
    n = np.floor((len(s) - FrameSize) / ShiftSize)  # 97
    loop_size = int(ShiftSize * n + FrameSize)  # 15920
    norm_t = norm(t, 2)  # 115.2325447
    for i in range(FrameSize, loop_size, ShiftSize):
        temp = np.log(norm(t[i - FrameSize:i], 2) / norm_t + 0.00001)
        # temp_all = np.insert(temp_all, temp)#[temp_all, temp]
        temp_all = np.hstack((temp_all, temp))
        if temp > threshold:
            # new = [new, 1 * np.ones(ShiftSize, 1)]
            new = np.hstack((new, 1 * np.ones(ShiftSize)))
        else:
            # new = [new, 0 * np.ones(ShiftSize, 1)]
            new = np.hstack((new, 0 * np.ones(ShiftSize)))

    # for i in range(ShiftSize * n + FrameSize):
    # for i in range(loop_size): #15920
    s_temp = np.array(s)

    end = len(new)  # len(s_temp)
    s_temp = s_temp[0:end]  # s_temp[0:(end - Overlap)]
    new_s = np.transpose(new) * s_temp

    for j in range(len(new)):
        if new[j] == 1:
            rest_s = np.hstack((rest_s, new_s[j]))
            # rest_s = np.insert(rest_s, new_s[j])
    return rest_s

def get_mel_fb(SR=16000, num_fft=1024, num_mels=40, F_Min=133, F_Max=1200):
    mel_fb = librosa.filters.mel(sr=SR, n_fft=num_fft, n_mels=num_mels, fmin=F_Min, fmax=F_Max, norm=None)
    ret_fb = mel_fb.T
    return ret_fb


def get_mfcc_librosa(wav_sig=None, sample_rate=16000, frame_length=400, step_length=160,
                     num_mels=40, num_mfccs=40, mel_fb=None, dct_type=2, window='hamming'):
    tmp_melspec = librosa.feature.melspectrogram(y=wav_sig, sr=sample_rate,
                                                 S=mel_fb,
                                                 n_mels=num_mels,
                                                 n_fft=1024,
                                                 hop_length=step_length,
                                                 win_length=frame_length,
                                                 window=window)

    tmp_melspec = librosa.power_to_db(tmp_melspec)
    _mfcc = librosa.feature.mfcc(S=tmp_melspec, dct_type=dct_type, n_mfcc=num_mfccs, norm=None, lifter=0)
    return _mfcc

def get_librosa_defult_mfcc(wav_sig=None, sample_rate=16000, frame_length=400, step_length=160,
                            num_mels=40, num_mfccs=40):
    mfcc_ = librosa.feature.mfcc(y=wav_sig, sr=16000, n_mfcc=40, n_mels=40,
                                 win_length=frame_length, hop_length=step_length)
    return mfcc_

def gen_train_data(sig_, label, melfb, sr_ = 16000):
    removed_sig = vad_test(sig_, sr_)
    len_of_sig_ = len(removed_sig)
    half_of_sr_len = int(sr_//2)
    if len_of_sig_ > half_of_sr_len:
        removed_sig = removed_sig[(len_of_sig_ - half_of_sr_len):len_of_sig_]
    elif len_of_sig_ < half_of_sr_len:
        pad_array = 0.5 + np.random.rand(half_of_sr_len - len_of_sig_) * 10 ** -6
        removed_sig = np.hstack((pad_array, removed_sig))

    coeff = get_mfcc_librosa(wav_sig=removed_sig, mel_fb=melfb, window=None)
    # coeff = get_librosa_defult_mfcc(wav_sig=removed_sig)
    temp = coeff[0,:]-np.amin(coeff[0,:])
    coeff[0, :] = temp / np.amax(temp)
    nframe = len(coeff[0, :])
    #[zeros(nDims, context_l), coeff, zeros(nDims, context_r)];
    coeff = np.hstack((np.zeros((nDims, context_l)), coeff))
    coeff = np.hstack((coeff, np.zeros((nDims, context_r))))
    x = np.zeros((nDims,0))
    y = np.zeros(0)
    for context in range(nframe):
        xx = np.zeros((0,40))
        window = coeff[:, context:(context+context_l+context_r)]
        wLoop = context_l+context_r
        for w in range(wLoop):
            be_stacked_win = window[:, w]
            xx = np.vstack((xx, window[:, w]))
        # xx = xx[1:]
        x = np.hstack((x, xx))
        y = np.hstack((y, label))
    return x, y