from lcj_io import getDirsInFolder, getFilesInFloder
import numpy


def preemphasis(signal, coeff=0.95):
    return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])

def hz2mel_nature(freq):
    return 1127. * numpy.log(1. + freq / 700.)

def mel2hz_nature(mel):
    return 700. * (numpy.exp(mel / 1127.) - 1.)

def hz2mel(hz):
    return 2595 * numpy.log10(1+hz/700.)

def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

# def framesig(sig, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,)), stride_trick=False):
def framesig(sig, frame_len, frame_step, stride_trick=False):
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = numpy.zeros((padlen - slen,))
    padsignal = numpy.concatenate((sig, zeros))
    
    indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
        numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    frames = padsignal[indices]
    return  frames

def magspec(frames, NFFT):
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
    numpy.shape(frames)[1], NFFT)
    complex_spec = numpy.fft.fft(frames, NFFT)
    return numpy.absolute(complex_spec)

def powspec(frames, NFFT):
    return numpy.square(frames)
#     return 1.0 / NFFT * numpy.square(frames)

def get_filterbanks(nfilt=10,nfft=1024,samplerate=16000,lowfreq=0,highfreq=8000):
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"
    # compute points evenly spaced in mels
    lowmel = hz2mel_nature(lowfreq)
    highmel = hz2mel_nature(highfreq)
    melpoints = numpy.linspace(lowmel,highmel,nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    mid_freqs = mel2hz_nature(melpoints)
    
    bins = numpy.floor((nfft+1)*mid_freqs/samplerate)
    fbank = numpy.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bins[j]), int(bins[j+1])):
            fbank[j,i] = (i - bins[j]) / (bins[j+1]-bins[j])
        for i in range(int(bins[j+1]), int(bins[j+2])):
            fbank[j,i] = (bins[j+2]-i) / (bins[j+2]-bins[j+1])
    print("Middel Frequences are {}".format(mid_freqs))
    print("Bins are {}".format(bins))
    return fbank
