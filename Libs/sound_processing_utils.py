import numpy as np
import random
import subprocess
#Fixed seed for reproduci
random.seed(42)
# Default data augmentation
def padding(pad):
    def f(sound):
        return np.pad(sound, pad, 'constant')

    return f


def random_crop(size):
    def f(sound):
        org_size = len(sound)
        start = random.randint(0, org_size - size)
        return sound[start: start + size]

    return f


def normalize(factor):
    def f(sound):
        return sound / factor

    return f

def rms_normalize(rms_level=0):
    def f(sig):
        r = 10**(rms_level / 10.0)
        a = np.sqrt( (len(sig) * r**2) / np.sum(sig**2) )
        # normalize
        y = sig * a
        return y
    return f

def minmax_normalize():
    def f(sig):
        X_min = np.min(sig)
        X_max = np.max(sig)
        X_norm = ((sig - X_min) / (X_max - X_min)) if  (X_max - X_min) > 0 else 0
        return X_norm
    return f

#########################################################
    # construct file names
    # output_file_path = os.path.dirname(infile)
    # name_attribute = "output_file.wav"

    # export data to file
    # write_file(output_file_path=output_file_path,
    #            input_file_name=infile,
    #            name_attribute=name_attribute,
    #            sig=y,
    #            fs=fs)


def convert_sr_for_dir(src_path, dst_path, sr):
    print('* {} -> {}'.format(src_path, dst_path))
    if not os.path.exists(dst_path):
        os.mkdir(dst_path);
    for src_file in sorted(glob.glob(os.path.join(src_path, '*.wav'))):
        dst_file = src_file.replace(src_path, dst_path);
        subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(
            src_file, sr, dst_file), shell=True);

def convert_sr_for_single_file(src_file, dst_file, sr):
    print('* {} -> {}'.format(src_file, dst_file))
    subprocess.call('ffmpeg -i {} -ac 1 -ar {} -loglevel error -y {}'.format(
        src_file, sr, dst_file), shell=True);

# For strong data augmentation
def random_scale(max_scale, interpolate='Linear'):
    def f(sound):
        scale = np.power(max_scale, random.uniform(-1, 1))
        output_size = int(len(sound) * scale)
        ref = np.arange(output_size) / scale
        if interpolate == 'Linear':
            ref1 = ref.astype(np.int32)
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = ref - ref1
            scaled_sound = sound[ref1] * (1 - r) + sound[ref2] * r
        elif interpolate == 'Nearest':
            scaled_sound = sound[ref.astype(np.int32)]
        else:
            raise Exception('Invalid interpolation mode {}'.format(interpolate))

        return scaled_sound

    return f


def random_gain(db):
    def f(sound):
        return sound * np.power(10, random.uniform(-db, db) / 20.0)

    return f


# For testing phase
# original codes of multi_crop
def multi_crop(input_length, n_crops):
    def f(sound):
        stride = (len(sound) - input_length) // (n_crops - 1)
        sounds = [sound[stride * i: stride * i + input_length] for i in range(n_crops)]
        return np.array(sounds)
    return f

# def multi_crop(input_length, n_crops):
#     def f(sound):
#         stride = (len(sound) - input_length) // (n_crops - 1)
#         print(f"strind = {stride}")
#         for i in range(n_crops):
#             clip_start = stride * i;
#             clip_end = stride * i + input_length;
#             print(f"clip-{i+1} range: {clip_start} to {clip_end}");
#         sounds = [sound[stride * i: stride * i + input_length] for i in range(n_crops)]
#         # print(f"sounds:{sounds}")
#         return np.array(sounds)
#     return f



#modify for only-one crop
def single_crop(input_length):
    def f(sound):
        stride = (len(sound) - input_length)
        sounds = [sound[stride + input_length]];
        # sounds = [sound[stride * i: stride * i + input_length] for i in range(n_crops)]
        return np.array(sounds)
    return f

# For BC learning
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 16000 or fs == 20000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    #no xrange anymore supported
    for i in range(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])
            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    return sound

# Convert time representation
def to_hms(time):
    h = int(time // 3600)
    m = int((time - h * 3600) // 60)
    s = int(time - h * 3600 - m * 60)
    if h > 0:
        line = '{}h{:02d}m'.format(h, m)
    else:
        line = '{}m{:02d}s'.format(m, s)

    return line
