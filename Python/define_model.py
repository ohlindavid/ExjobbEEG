import tensorflow as tf
from MorletLayer import MorletConv, MorletConvRaw
from tensorflow.keras import layers, optimizers, losses, Input, regularizers
#from tensorflow_addons import metrics.CohenKappa
import datetime
from settings import etas, filters, wtime

def define_Morlet(nchan,L,Fs):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((L,nchan),batch_size=None))
    model.add(layers.LayerNormalization(axis=[-1]))
    model.add(MorletConvRaw([L,nchan],Fs,input_shape=[L,nchan,1],etas=etas,wtime=wtime))
    model.add(layers.Conv2D(filters=filters, kernel_size=[1,nchan], activation='elu'))
    model.add(layers.Permute((3,1,2), name="second_permute"))
    model.add(layers.AveragePooling2D(pool_size=(1,71), strides=(1,15),name="pooling"))
    model.add(layers.Dropout(0.75))
    model.add(layers.Flatten())
    model.add(layers.Dense(3))
    model.add(layers.Activation('softmax'))
    model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=optimizers.Adam(lr=0.0001),
        metrics=['accuracy'],
        run_eagerly = True)
    return model

def define_1DCNN(nchan,L,Fs):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((L,nchan),batch_size=None))
    model.add(layers.Conv1D(filters=30, kernel_size=64,padding="causal"))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.AveragePooling1D(pool_size=(2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv1D(filters=15, kernel_size=32,padding="causal"))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.AveragePooling1D(pool_size=(2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv1D(filters=10, kernel_size=16,padding="causal"))
    model.add(layers.LayerNormalization())
    model.add(layers.Activation('elu'))
    model.add(layers.AveragePooling1D(pool_size=(2)))
    model.add(layers.Dropout(0.4))
    model.add(layers.Flatten())
    model.add(layers.Dense(15,activation="tanh"))
    model.add(layers.LayerNormalization())
    model.add(layers.Dense(3))
    model.add(layers.Activation('softmax'))
    model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=optimizers.Adam(),
        metrics=['accuracy'],
        run_eagerly = False)
    return model

def define_2DR(nchan,L,Fs,batch_size=None):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((Fs, L, nchan),batch_size=None))
    model.add(layers.LayerNormalization(axis=[-1],center=True,scale=True))
    model.add(layers.Conv2D(filters=4, kernel_size=[8,4], activation='elu'))
    model.add(layers.AveragePooling2D(pool_size=(4, 4)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(filters=8, kernel_size=[8,4], activation='elu'))
    model.add(layers.AveragePooling2D(pool_size=(4, 4)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(filters=16, kernel_size=[8,4], activation='elu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(15, activation='tanh',kernel_regularizer=regularizers.l1(l1=1e-2)))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=optimizers.Adam(lr=0.0001),
        metrics=['accuracy'],
        run_eagerly = False)
    return model

def define_RecR(nchan,L,Fs):
    model = tf.keras.Sequential()
    model.add(layers.InputLayer((L,nchan),batch_size=None))
    model.add(layers.LayerNormalization(axis=[-1]))
    model.add(MorletConvRaw([L,nchan],Fs,input_shape=[L,nchan,1],etas=etas,wtime=wtime))
    model.add(layers.Conv2D(filters=filters, kernel_size=[1,nchan], activation='elu'))
    model.add(layers.Permute((3,1,2), name="second_permute"))
    model.add(layers.AveragePooling2D(pool_size=(1,15), strides=(1,4),name="pooling"))
    model.add(layers.Dropout(0.75))
    model.add(layers.Flatten())
    model.add(layers.Dense(3))
    model.add(layers.Activation('softmax'))
    model.compile(
        loss=losses.CategoricalCrossentropy(),
        optimizer=optimizers.Adam(),
        metrics=['accuracy'],
        run_eagerly = False)
    return model

def load_tensorboard(who,date,fold):
    if (who=="Oskar"):
        log_dir = "C:/Users/Oskar/Documents/GitHub/Exjobb/logs/fit/" + str(date) + "/" + str(fold+1)
    else:
        log_dir = "C:/Users/Kioskar/Documents/GitHub/Exjobb/logs/fit/" + str(date) + "/" + str(fold+1)
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
