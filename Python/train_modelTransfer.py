import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Input
import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from split_data import split_data
from settings import checkpoint_path
from generate_CAN import generate_gradCAM
import matplotlib.pyplot as plt
import glob
np.set_printoptions(threshold=sys.maxsize)

def train_model(spex,subject,date):
	gpus = tensorflow.config.experimental.list_physical_devices('GPU')
	tensorflow.config.experimental.set_memory_growth(gpus[0], True)
	k_folds = 10
	(define_model,epochs,L,Fs,nchan,modelName) = spex

	path = "C:/Users/Kioskar/Desktop/Testing exjobb/Albin_Damir/AD_crop/"+ modelName + "/"
	names = glob.glob(path+'*/*', recursive=True) #os.listdir(path)
	names = [ x for x in names if "subj" + str(subject) not in x ]
	#print(names)
	np.random.shuffle(names)

	vals = []
	#map = np.zeros([3,120])
	for i in range(0,1):
		print("Fold number " + str(i+1) + "!")
		(gen,genVal,trainlen,vallen) = split_data(["A","B","C"],k_folds,i,names,path,spex,class_on_char=74)

		checkpoint_path_fold = checkpoint_path + date + "/fold" + str(i+1) + "/cp-{epoch:04d}.ckpt"
		cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_fold,save_weights_only=True,verbose=1)

		model = define_model(nchan,L,Fs)
		history = model.fit(
			gen,
			validation_data=genVal,
			steps_per_epoch=trainlen,
			validation_steps=vallen,
			epochs=30,
			callbacks=[cp_callback],
			verbose=2)
		#heatmap_mean = generate_gradCAM(model,spex,path,gen,trainlen)
		#plt.imshow(np.repeat(heatmap_mean,50,axis=0))
		#plt.show()
		#vals.append(history.history['accuracy'])
	return vals
