from os import listdir
from os.path import isfile, join, isdir

def getFilesInFloder(folderPath):
    onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
    return onlyfiles

def getDirsInFolder(baseDirPath):
    onlySubDirs = [d for d in listdir(baseDirPath) if isdir(join(baseDirPath, d))]
    return onlySubDirs

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