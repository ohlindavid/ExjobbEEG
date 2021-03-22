import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Input
from MorletLayer import MorletConv
from tensorflow.keras.callbacks import EarlyStopping
from plots import show_loss, show_accuracy
import matplotlib.pyplot as plt
import os,sys
import tensorflow.keras.backend as K
from settings import who, epochs, checkpoint_path
from generator import signalLoader
from define_model import define_model, load_tensorboard
from split_data import split_data
import math
import datetime
gpus = tensorflow.config.experimental.list_physical_devices('GPU')
tensorflow.config.experimental.set_memory_growth(gpus[0], True)
batch_size = 8

def path():
    if who=="Oskar":
        return "C:/Users/Kioskar/Documents/GitHub/exjobb/Testing Sets/sets/Albin&Damir/study_all_subjects/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/AD_retrieval_transfer_crop_subject_1/"
def pathPred():
    if who=="Oskar":
        return "C:/Users/Oskar/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"
    if who=="David":
        return "C:/Users/david/Documents/GitHub/exjobb/Testing Sets/Albin&Damir/AD_data_set_subject_6/"

nchan = 31 #Antal kanaler
L = 2049 #EEG-längd per epok innan TF-analys
Fs = 512
data_aug = False
doDownsampling = False

allnames = os.listdir(path())

subjectnames = np.unique([i.split("_")[0] for i in allnames])
subjectnames = np.unique([e[1:] for e in subjectnames])
#np.random.shuffle(subjectnames)

val_accs_all_subj = []
for subject in subjectnames:
    names = []
    for i,name in enumerate(allnames):
        if name[0:13] == ('A'+ subject):
            names.append(name)
        if name[0:13] == ('B'+ subject):
            names.append(name)
        if name[0:13] == ('C'+ subject):
            names.append(name)

    k_folds= 10
    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    val_accs = []
    for i in range(0,k_folds):
        print("Fold number " + str(i+1) + "!")

        (data_generator,data_generatorVal,l,lv) = split_data(['A','B','C'],k_folds,i,names,path(),nchan,data_aug,batch_size=batch_size)

        tensorboard_callback = load_tensorboard(who,date,i)
        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
        checkpoint_path_fold = checkpoint_path + date + "/fold" + str(i+1) + "/cp-{epoch:04d}.ckpt"
        check_point_dir = os.path.dirname(checkpoint_path_fold)
        cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_fold,save_weights_only=True,verbose=1)

        model = define_model(nchan,L,Fs,batch_size=batch_size)
    # Load weights:
        #model.load_weights("C:/Users/Oskar/Documents/GitHub/Exjobb/logs/model_check_points/20210126-143212/fold1/cp-0005.ckpt")
        #model.trainable = False  # Freeze the outer model
        tensorflow.executing_eagerly()
        history = model.fit(data_generator,batch_size = batch_size,validation_data=data_generatorVal,steps_per_epoch=l/batch_size-1,validation_steps=lv/batch_size-1,epochs=epochs,callbacks=[tensorboard_callback,cp_callback],verbose=1)
        model.summary()
        val_accs.append(np.max(history.history['val_accuracy']))
        print(val_accs)
        print(np.mean(val_accs))
    val_accs_all_subj.append(np.mean(val_accs))
    print(val_accs_all_subj)
print(np.mean(val_accs_all_subj))
