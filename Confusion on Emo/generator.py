import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
import numpy as np
import random
from scipy.signal import decimate
import matplotlib.pyplot as plt

def signalLoader(files,path,spex,batch_size=1,class_on_char=0):

    if class_on_char > 0:
        path = ""
    nfiles = len(files)
    np.random.shuffle(files)
    X = methodToLoad(files,path,spex,batch_size=nfiles)

    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < nfiles:
            limit = min(batch_end, nfiles)
            Y = np.zeros([batch_size,3])
            for i,name in enumerate(files[batch_start:limit]):
                if name[class_on_char] == ('A'):
                    Y[i,:] = [1,0,0]
                if name[class_on_char] == ('B'):
                    Y[i,:] = [0,1,0]
                if name[class_on_char] == ('C'):
                    Y[i,:] = [0,0,1]
            #plt.imshow(files_array[batch_start,:,:,0])
            #plt.show()
            #print(np.shape(X[batch_start:limit,:,:]),Y)
            yield (X[batch_start:limit,:,:],Y)
            batch_start += batch_size
            batch_end += batch_size

def methodToLoad(files,path,spex,batch_size=1):
    (_,_,L,Fs,nchan,modelName) = spex
    if modelName == "Reassignment2D" or modelName == "Spectrogram2D":
        train_0 = np.zeros([batch_size,Fs,L,nchan])
        for i,imID in enumerate(files):
            spec = np.loadtxt(path+imID,delimiter=',')
            spec = np.reshape(spec,[nchan,L,Fs])
            spec = np.transpose(spec,[2,1,0])
            train_0[i,:,:,:] = spec
        return train_0
    else:
        train_0 = np.zeros([batch_size,L,nchan])
        for i,imID in enumerate(files):
            train_0[i,:,:] = np.loadtxt(path+imID,delimiter=',')
        return train_0
