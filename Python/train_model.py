import tensorflow
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Input
import os,sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from generator import signalLoader
from define_model import load_tensorboard
from split_data import split_data
from settings import checkpoint_path
from generate_CAN import generate_gradCAM
import matplotlib.pyplot as plt
import sys
from pympler.tracker import SummaryTracker

def train_model(spex,subject,date,pretrain = False):
	batch_size = 1
	k_folds = 10
	(define_model,epochs,L,Fs,nchan,modelName) = spex

	path = "C:/Users/Kioskar/Desktop/Testing exjobb/Albin_Damir/all_subj_crop/" + modelName + "/subj" + str(subject) + "/"
	#path2 = "C:/Users/Kioskar/Desktop/Testing exjobb/Albin_Damir/all_subj_crop/" + modelName + "/subj" + str(subject) + "/"
	names = os.listdir(path)


#	classes = ["A","B","C"]
	k = k_folds
#	list_names = [[],[],[]]
#	i = 0
#	for c in classes:
#		the_names = [idx for idx in names if idx[0].lower() == c.lower()]
#		list_names[i] = np.array_split(the_names,k)
#		i = i + 1
#	def methodToLoad(files,path,spex,batch_size=1):
#		(_,_,L,Fs,nchan,modelName) = spex
#		train_0 = np.zeros([batch_size,Fs,L,nchan])
#		for i,imID in enumerate(files):
#			spec = np.loadtxt(path+imID,delimiter=',')
#			spec = np.reshape(spec,[nchan,L,Fs])
#			spec = np.transpose(spec,[2,1,0])
#			train_0[i,:,:,:] = spec
#		return train_0
#	files  = [[],[],[]]
#	for i in range(0,3):
#		for the_names in list_names[i]:
#			files[i].append(methodToLoad(the_names,path,spex,batch_size=len(the_names)))

	np.random.shuffle(names)

	vals = []
	#tracker = SummaryTracker()
	#vals_pretrain = []
	confusion_matrix = np.zeros([k_folds,3,3])
	for i in range(0,k_folds):
		gpus = tensorflow.config.experimental.list_physical_devices('GPU')
		tensorflow.config.experimental.set_memory_growth(gpus[0], True)

		print("Fold number " + str(i+1) + "!")
		#fold_files = []
		#fold_files_val = []
		#class_names = []
		#class_names_val = []
		#for j in range(0,3):
		#	class_names.extend(np.hstack(np.delete(list_names[j], i, 0)).transpose())
		#	class_names_val.extend(list_names[j][i])
		#	fold_files.extend(np.delete(files[j], i, 0))
		#	fold_files_val.extend(files[j])
		#fold_files = np.vstack(fold_files)
		#fold_files_val = np.vstack(fold_files_val)
		#genVal = signalLoader(class_names_val,fold_files_val,path,spex,batch_size=batch_size,class_on_char=0)
		#gen = signalLoader(class_names,fold_files,path,spex,batch_size=batch_size,class_on_char=0)
		#print(class_names)
		#print(class_names_val)
		#trainlen = len(class_names)
		#vallen = len(class_names_val)
		#print(vallen)
		(gen,genVal,trainlen,vallen) = split_data(["A","B","C"],k_folds,i,names,path,spex,batch_size=batch_size)
		checkpoint_path_fold = checkpoint_path + date + "/fold" + str(i+1) + "/cp-{epoch:04d}.ckpt"
			#cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path_fold,save_weights_only=True,verbose=1)
		model = define_model(nchan,L,Fs)
			#model.load_weights("C:/Users/Kioskar/Documents/GitHub/Exjobb/logs/model_check_points/20210311-183357/fold1/cp-0042.ckpt") #Subj03
			#model.load_weights("C:/Users/Kioskar/Documents/GitHub/Exjobb/logs/model_check_points/20210312-095405/fold1/cp-0051.ckpt") #Subj01
			#if pretrain :
			#	model.load_weights("C:/Users/Kioskar/Documents/GitHub/Exjobb/logs/model_check_points/"+date+"/fold1/cp-0030.ckpt") #for transfer algorithm.
			#history_pretrain = model.evaluate(gen,steps=trainlen,verbose=2)
			#print(history_pretrain)
			#vals_pretrain.append(history_pretrain[1])
		history = model.fit(
			gen,
			validation_data=genVal,
			steps_per_epoch=int(trainlen/batch_size)+1,
			validation_steps=int(vallen/batch_size)+1,
			epochs=epochs,
			#callbacks=[cp_callback],
			verbose=2)
		vals.append(history.history['val_accuracy'])
		del history
		del model
		tensorflow.keras.backend.clear_session()
		tensorflow.compat.v1.reset_default_graph()
		def limit_mem():
			tensorflow.config.experimental.get_session().close()
		#limit_mem()
			#heatmap_mean = generate_gradCAM(model,spex,path)


			#plt.imshow(np.repeat(heatmap_mean,50,aixs=0))
			#plt.show()
			#labels = np.zeros([vallen,3])
			#for j in range(0,vallen):
			#	(_,labels[j,:]) = next(genVal)
			#val_preds = np.argmax(model.predict(genVal,steps=vallen),axis=1)
			#confusion_matrix[i,:,:] = tensorflow.math.confusion_matrix(np.argmax(labels,axis=1),val_preds)
			#print(confusion_matrix[i,:,:])
			#names2 = os.listdir(path2)
			#(gen2,_,trainlen2,_) = split_data(["A","B","C"],100,0,names2,path2,spex,batch_size=batch_size)
			#history2 = model.evaluate(gen2,steps=trainlen2)
			#print(history2)
#pri	nt(np.mean(confusion_matrix,axis=0))
#pri	nt(vals_pretrain)
#pri	nt(np.mean(vals_pretrain))
		#del fold_files
		#del fold_files_val
		#tracker.print_diff()
	return vals
