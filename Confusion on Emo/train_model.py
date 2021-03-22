import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Input
import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from generator import signalLoader
from define_model import load_tensorboard
from split_data import split_data
from settings import checkpoint_path
#from generate_CAN import generate_gradCAM
import matplotlib.pyplot as plt
import sys
from pympler.tracker import SummaryTracker

def train_model(spex,subject,date,pretrain = False):
	batch_size = 1
	k_folds = 10
	(define_model,epochs,L,Fs,nchan,modelName) = spex

	path = "C:/Users/Kioskar/Desktop/Testing exjobb/EmoDecode1/Study/" + modelName + "/subj" + str(subject) + "/"
	path2 = "C:/Users/Kioskar/Desktop/Testing exjobb/EmoDecode1/Retrieval/" + modelName + "/subj" + str(subject) + "/"
	names = os.listdir(path)
	np.random.shuffle(names)

	vals = []
	confusion_matrix_F = np.zeros([k_folds,3,3])
	confusion_matrix_S = np.zeros([k_folds,3,3])


	for i in range(0,k_folds):
		gpus = tensorflow.config.experimental.list_physical_devices('GPU')
		print(gpus)
		print(tensorflow.__version__)
		tensorflow.config.experimental.set_memory_growth(gpus[0], True)
		print(tensorflow.config.experimental.get_memory_growth(gpus[0]))

		print("Fold number " + str(i+1) + "!")

		(gen,genVal,trainlen,vallen,val_names) = split_data(["A","B","C"],k_folds,i,names,path,spex,batch_size=batch_size)
		checkpoint_path_fold = checkpoint_path + date + "/fold" + str(i+1) + "/cp-{epoch:04d}.ckpt"
		#cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_fold,save_weights_only=True,verbose=1)
		model = define_model(nchan,L,Fs)
		#model.load_weights("C:/Users/Kioskar/Documents/GitHub/Exjobb/logs/model_check_points/20210311-183357/fold1/cp-0042.ckpt") #Subj03
		history = model.fit(
			gen,
			validation_data=genVal,
			steps_per_epoch=int(trainlen/batch_size)+1,
			validation_steps=int(vallen/batch_size)+1,
			epochs=epochs,
			#callbacks=[cp_callback],
			verbose=2)
		vals.append(history.history['val_accuracy'])

		#heatmap_mean = generate_gradCAM(model,spex,path)
		#plt.imshow(np.repeat(heatmap_mean,50,aixs=0))
		#plt.show()
		labels = np.zeros([vallen,3])
		labels_FS = np.zeros(vallen)
		for j in range(0,vallen):
			(_,labels[j,:]) = next(genVal)
			if val_names[j][4] == "F":
				labels_FS[j] = 1
		print(labels_FS)
		val_preds = np.argmax(model.predict(genVal,steps=vallen),axis=1)
		confusion_matrix_F[i,:,:] = tensorflow.math.confusion_matrix(np.argmax(labels,axis=1)[labels_FS==1],val_preds[labels_FS==1])
		print(np.argmax(labels,axis=1)[labels_FS==0])
		print(val_preds[labels_FS==0])
		confusion_matrix_S[i,:,:] = tensorflow.math.confusion_matrix(np.argmax(labels,axis=1)[labels_FS==0],val_preds[labels_FS==0])

		print(confusion_matrix_F[i,:,:])
		print(confusion_matrix_S[i,:,:])
		names2 = os.listdir(path2)
		(gen2,_,trainlen2,_,_) = split_data(["A","B","C"],100,0,names2,path2,spex,batch_size=batch_size)
		history2 = model.evaluate(gen2,steps=trainlen2)
		print(history2)





		del history
		del model
		tensorflow.keras.backend.clear_session()
		tensorflow.compat.v1.reset_default_graph()
		def limit_mem():
			tensorflow.config.experimental.get_session().close()
		#limit_mem()
	print("final conf")
	print(np.mean(confusion_matrix_F,axis=0))
	print(np.mean(confusion_matrix_S,axis=0))
	return vals
