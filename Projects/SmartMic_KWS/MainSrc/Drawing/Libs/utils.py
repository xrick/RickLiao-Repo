# import random
# import itertools
# import librosa
# import matplotlib.pyplot as plt
# import os
# import soundfile as sf
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import wave
import struct
from tqdm import tqdm
import subprocess
import re
import os
import struct

min_value = np.iinfo('int16').min
max_value = np.iinfo('int16').max
ffmpeg_mp3_to_wav_cmd = "ffmpeg -i {mp3} -vn -acodec pcm_s16le -ac 1 -ar {sr} -f wav {wav}"
mpg123_mp3_to_wav_cmd = "mpg123 -w {wav} {mp3}"

def getFilesInFloder(folderPath):
    onlyfiles = [f for f in listdir(folderPath) if isfile(join(folderPath, f))]
    return onlyfiles

def getDirsInFolder(baseDirPath):
    onlySubDirs = [d for d in listdir(baseDirPath) if isdir(join(baseDirPath, d))]
    return onlySubDirs

def get_recursive_files(folderPath, regex='.*\.wav'):
    results = os.listdir(folderPath)
    out_files = []
    cnt_files = 0
    for file in results:
        if os.path.isdir(os.path.join(folderPath, file)):
            out_files += get_recursive_files(os.path.join(folderPath, file), regex)
        elif re.match(regex, file,  re.I):  # file.startswith(startExtension) or file.endswith(".txt") or file.endswith(endExtension):
            out_files.append(os.path.join(folderPath, file))
            cnt_files = cnt_files + 1
    return out_files

def Dat2WAV(files, target_path):
    sampleRate = 16000
    for file in tqdm(files):
        # read dat file
        if file.endswith(".dat"):
            tmp_list = []
            with open(target_path + file, "r") as rf:
                tmp_list = rf.readlines()

            print("file:{} length is {}".format(file, len(tmp_list)))
            with wave.open(target_path + "wav/{}.wav".format(file), "w")as obj:
                obj.setnchannels(1)  # mono
                obj.setsampwidth(2)
                obj.setframerate(sampleRate)
                tmp_list = tmp_list[1:]
                for value in tmp_list:
                    value = value.replace("\n", "")
                    value = int(float(value))
                    data = struct.pack('<h', value)
                    obj.writeframesraw(data)
    print("processing completed.")

def run_win_cmd(cmd):
    result = []
    process = subprocess.Popen(cmd,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    for line in process.stdout:
        result.append(line)
    errcode = process.returncode
    for line in result:
        print(line)
    if errcode is not None:
        raise Exception('cmd %s failed, see above for details', cmd)

def add_white_noise(data):
    wn = np.random.randn(len(data))
    data_wn = data + 0.005 * wn
    return data_wn

def convert_mp3_to_wav(speech_mp3_files_path, target_path):
    speech_mp3_files_list = getFilesInFloder(speech_mp3_files_path)
    for f in tqdm(speech_mp3_files_list):
        command = ffmpeg_mp3_to_wav_cmd.format(mp3=speech_mp3_files_path+f, sr=16000, wav=target_path.format(f))
        os.system(command)
        # "./TestWavFiles/Speech/wav/{}.wav"

if __name__ == "__main__":
    srcDir1 = "."