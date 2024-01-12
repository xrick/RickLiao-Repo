import os
import sys
from os import listdir
from os.path import isfile, join
import re

def getFileList(srcDir,regex='.*\.wav'):
    results = os.listdir(srcDir)
    out_files = []
    cnt_files = 0
    for file in results:
        if os.path.isdir(os.path.join(srcDir, file)):
            out_files += getFileList(os.path.join(srcDir, file))
        elif re.match(regex, file,  re.I):  # file.startswith(startExtension) or file.endswith(".txt") or file.endswith(endExtension):
            out_files.append(os.path.join(srcDir, file))
            cnt_files = cnt_files + 1
    return out_files


def getFilesInFloder(folderPath):
    onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
    return onlyfiles

def getDirsInFolder(baseDirPath):
    onlySubDirs = [d for d in listdir(baseDirPath) if isdir(join(baseDirPath, d))]
    return onlySubDirs

def safe_wav_read(wav_file, sr):
    try:
        std_sr = 16000
        sr, sig = wavio.read(wav_file)
        if sig.shape[0] < sig.size:
            sig = sig[0]
            print("\n{} is channel 2".format(wav_file))
        return sr, sig
    except:
        print("Error occured in read and convert wav to ndarray in file {}".format(wav_file))