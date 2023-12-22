import sys
import os
import numpy as np
import scipy.io.wavfile as wavio
import logging
import decimal
import math
from scipy.fftpack import dct
from scipy.io import wavfile
# def getFilesInFloder(folderPath):
#     onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
#     return onlyfiles
#
# def getDirsInFolder(baseDirPath):
#     onlySubDirs = [d for d in listdir(baseDirPath) if isdir(join(baseDirPath, d))]
#     return onlySubDirs

def safe_wav_read(wav_file):
    try:
        std_sr = 16000
        sr, sig = wavio.read(wav_file)
        if sig.shape[0] < sig.size:
            sig = sig[0]
            print("\n{} is channel 2".format(wav_file))
        return sr, sig
    except:
        print("Error occured in read and convert wav to ndarray in file {}".format(wav_file))

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
            np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = np.array(indices, dtype=np.int32)
        frames = padsignal[indices]
        win = np.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def hz2mel_nature(freq):
    return 1127. * np.log(1. + freq / 700.)

def mel2hz_nature(mel):
    return 700. * (np.exp(mel / 1127.) - 1.)

def hz2mel(hz):
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)


# def get_filterbanks_from_40(nfilt=40, nfft=1024, samplerate=16000, lowfreq=70, highfreq=8000):
#     highfreq = highfreq or samplerate / 2
#     assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"
#
#     # compute points evenly spaced in mels
#     lowmel = hz2mel_nature(lowfreq)
#     highmel = hz2mel_nature(highfreq)
#
#     melpoints = np.linspace(lowmel, highmel, nfilt + 2)
#     mid_freqs = mel2hz_nature(melpoints)
#     # *********************************************
#     c = 0
#     for freq in mid_freqs:
#         print("{}-{}".format("get_filterbanks_from_40", freq))
#         c += 1
#     target_mid_freqs = np.empty(12, dtype=np.float)
#     idx = 0
#     for i in (2, 3, 5, 6, 8, 9, 10, 12, 22, 32):
#         print(mid_freqs[i])
#         target_mid_freqs[idx] = mid_freqs[i]
#         idx += 1
#     nfilt = 10
#     # *********************************************
#     bin = np.floor((nfft + 1) * target_mid_freqs / samplerate)
#     fbank = np.zeros([nfilt, nfft // 2 + 1])
#     for j in range(0, nfilt):
#         for i in range(int(bin[j]), int(bin[j + 1])):
#             fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
#         for i in range(int(bin[j + 1]), int(bin[j + 2])):
#             fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])
#     return fbank

def get_filterbank_from_midfreqs(midFreqs,samplerate=16000, n_filt=10, n_fft=1024):
#     mid_freqs = midFreqs#[229.8,304.1,402.4,532.4,704.4,931.9,1233.1,1631.5,4000.,5500.]
    target_mid_freqs = np.empty(n_filt+2,dtype=np.float)
    idx = 0
    for freq in midFreqs:
        target_mid_freqs[idx] = freq
        idx += 1
#     target_mid_freqs[n_filt]=0.0
#     target_mid_freqs[n_filt+1]=0.0
    print(target_mid_freqs)
    bins = np.floor((n_fft+1)*target_mid_freqs/samplerate)
    print(len(bins))
    fbank = np.zeros([n_filt,n_fft//2+1])
    for j in range(0,n_filt):
        for i in range(int(bins[j]), int(bins[j+1])):
            fbank[j,i] = (i - bins[j]) / (bins[j+1]-bins[j])
        for i in range(int(bins[j+1]), int(bins[j+2])):
            fbank[j,i] = (bins[j+2]-i) / (bins[j+2]-bins[j+1])
    return fbank


def get_filterbanks(nfilt=40, nfft=1024, samplerate=16000, lowfreq=0, highfreq=8000):
    highfreq = highfreq or samplerate / 2
    """Compute log Mel-filterbank energy features from an audio signal.
    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the sample rate of the signal we are working with, in Hz.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use np window functions here e.g. winfunc=np.hamming
    :returns: A np array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    """
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"
    # compute points evenly spaced in mels
    lowmel = hz2mel_nature(lowfreq)
    highmel = hz2mel_nature(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    mid_freqs = mel2hz_nature(melpoints)

    bins = np.floor((nfft + 1) * mid_freqs / samplerate)
    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bins[j]), int(bins[j + 1])):
            fbank[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            fbank[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
    print("Middel Frequences are {}".format(mid_freqs))
    print("Bins are {}".format(bins))
    return fbank

def magspec(frames, NFFT):
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.fft(frames, NFFT)
#     complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)

def powspec(frames, NFFT):
    theFrames = magspec(frames,NFFT)
    return np.square(theFrames)
#     return 1.0 / NFFT * np.square(theFrames)

def get_mel_spectrum(sig=None, NFFT=1024, MelFB=None):
#     sr, sig = wavio.read(wavfile)
    sig_pspec = powspec(sig, NFFT)
    sig_pspec = np.split(sig_powspec.T,[0,(NFFT/2+1)],axis=0)[1]
    spec = np.matmul(MelFB, sig_pspec)
    spec = np.where(spec == 0, np.finfo(float).eps, spec)  # if feat is
    log_spec = np.log(spec)
    return log_spec


def dumpFB_Array(FB=None, midfreqs=None, fb_save_path=None):
#     theFB = get_filterbank_from_midfreqs(midfreqs, 16000, 10, 1024)
    rows = FB.shape[0]
    cols = FB.shape[1]
    with open(fb_save_path, "w")as f:
        if midfreqs is not None:
            f.write("/* middle frequences:{} */\n".format(midfreqs))
        f.write("const float fbarray[{}][{}]=".format(rows,cols))
        f.write("{")
        for i in range(rows):
            f.write("{")
            for j in range(cols):
                f.write(str(FB[i][j]))
                f.write(",")
            f.write("},")
            f.write("\n")
        f.write("}")

def framesig(sig, frame_len, frame_step, stride_trick=False):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))
    padlen = int((numframes - 1) * frame_step + frame_len)
    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))
    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
        np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = padsignal[indices]
    return  frames
        
def custom_mfcc_features(path_file, frame_size, frame_stride):
    sample_rate, signal = wavfile.read(path_file)
#     pre_emphasis = 0.97
#     emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # params
    '''frame_size = 0.025
    frame_stride = 0.01'''
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) +\
        np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    # hamming window
#     frames *= np.hamming(frame_length)

    NFFT = 1024
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB
    
    num_ceps = 20
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    
    cep_lifter = 22
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  #*
    
    return filter_banks, mfcc

# def mfcc_features(path_file, frame_size, frame_stride):
#     sample_rate, signal = wavfile.read(path_file)
#     pre_emphasis = 0.97
#     emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

#     # params
#     '''frame_size = 0.025
#     frame_stride = 0.01'''
#     frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
#     signal_length = len(emphasized_signal)
#     frame_length = int(round(frame_length))
#     frame_step = int(round(frame_step))
#     num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

#     pad_signal_length = num_frames * frame_step + frame_length
#     z = np.zeros((pad_signal_length - signal_length))
#     pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

#     indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) +\
#         np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
#     frames = pad_signal[indices.astype(np.int32, copy=False)]

#     # hamming window
# #     frames *= np.hamming(frame_length)

#     NFFT = 512
#     mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
#     pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

#     nfilt = 40
#     low_freq_mel = 0
#     high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
#     mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
#     hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
#     bin = np.floor((NFFT + 1) * hz_points / sample_rate)

#     fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
#     for m in range(1, nfilt + 1):
#         f_m_minus = int(bin[m - 1])   # left
#         f_m = int(bin[m])             # center
#         f_m_plus = int(bin[m + 1])    # right

#         for k in range(f_m_minus, f_m):
#             fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
#         for k in range(f_m, f_m_plus):
#             fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
#     filter_banks = np.dot(pow_frames, fbank.T)
#     filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
#     filter_banks = 20 * np.log10(filter_banks)  # dB
    
#     num_ceps = 20
#     mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    
#     cep_lifter = 22
#     (nframes, ncoeff) = mfcc.shape
#     n = np.arange(ncoeff)
#     lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
#     mfcc *= lift  #*
    
#     return filter_banks, mfcc
"""
middleFreq = [16, 20, 26, 36, 48, 60, 80, 101, 256, 353]
inband = 4
bandnum_ = 10
y_length_=513
"""
def genICFilterMatrix(bandnum, y_length, mid_freq_matrix):
    ret_icfilter = np.zeros((bandnum, y_length),dtype=float)
    for i in range(bandnum):
        if i == 8:
            for j in range(y_length):
                ret_icfilter[i][j]=10**((-360*abs(np.log10(15.625*j+1)-np.log10(15.625*(mid_freq_matrix[i]-1))))/20)
        elif i == 9:
            for j in range(y_length):
                ret_icfilter[i][j]=10**((-360*abs(np.log10(15.625*j+1)-np.log10(15.625*(mid_freq_matrix[i]-1))))/20)
        else:
            for j in range(y_length):
                ret_icfilter[i][j]=10**((-20*12*abs(np.log10(15.625*j+1)-np.log10(15.625*(mid_freq_matrix[i]-1))))/20)
    
    return ret_icfilter