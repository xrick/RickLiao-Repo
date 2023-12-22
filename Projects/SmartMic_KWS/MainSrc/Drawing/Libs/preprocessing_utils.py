import numpy as np

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def datachange(input):
    out = []
    for i in range(len(input)):
        if input[i]==1:
            out.append([1,0,0])
        elif input[i]==2:
            out.append([0,1,0])
        else:
            out.append([0,0,1])
    return out

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)