import os,sys,logging,datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
tensorflow.get_logger().setLevel(logging.ERROR)
from define_model import define_Morlet, define_2DR, define_RecR, define_1DCNN, load_tensorboard
from train_model import train_model
import numpy as np

#Directory: C:/Users/david/Documents/GitHub/exjobb/Testing Sets/final/subj(N)/(modellnamn)/(A/B/C)(Trialnummer)
n_subj = 18
models = {
#Tuple: (function,epochs,L,Fs,nchan)
#	"1DCNN":(define_1DCNN,40,820,512,62,"1DCNN"),
#	"Morlet":(define_Morlet,100,820,512,62,"Morlet"),
#	"ReassignmentRec":(define_RecR,30,193,256,31,"ReassignmentRec"),
#	"ReassignmentRecCorrect":(define_RecR,30,193,256,31,"ReassignmentRecCorrect"),
#	"SpectrogramRec":(define_RecR,30,193,256,31,"SpectrogramRec"),
#	"SpectrogramRecCorrect":(define_RecR,30,193,256,31,"SpectrogramRecCorrect"),
	"Spectrogram2D":(define_2DR,50,193,256,31,"Spectrogram2D"),
	"Reassignment2D":(define_2DR,50,193,256,31,"Reassignment2D")
}

histories = {
#	"1DCNN":np.zeros([models["1DCNN"][1],n_subj]),
#	"Morlet":np.zeros([models["Morlet"][1],n_subj]),
#	"ReassignmentRec":np.zeros([models["ReassignmentRec"][1],n_subj]),
#	"ReassignmentRecCorrect":np.zeros([models["ReassignmentRecCorrect"][1],n_subj]),
#	"SpectrogramRec":np.zeros([models["SpectrogramRec"][1],n_subj]),
#	"SpectrogramRecCorrect":np.zeros([models["SpectrogramRecCorrect"][1],n_subj]),
	"Spectrogram2D":np.zeros([models["Spectrogram2D"][1],n_subj]),
	"Reassignment2D":np.zeros([models["Reassignment2D"][1],n_subj])
}

for model in models:
	subjects = []
	for folder in os.listdir("C:/Users/Kioskar/Desktop/Testing exjobb/Albin_Damir/all_subj_crop/"+model+"/"):
		subjects.append(folder[4:6])
	print(subjects)
#	subjects = ["01","02"]
	print(subjects)
	print("Model: " + model + ", Epochs: " + str(models[model][1]))
	print("--------------------------------------------------------------------------")
	i = 0
	for subject in subjects:
		print("Subject: " + str(subject))
		date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		vals = np.array(train_model(models[model],subject,date))
		print(vals)
		histories[model][:,i] = np.mean(vals,axis=0)
		print(histories[model])
		i = i + 1
		print(np.sum(histories[model],axis=1)/np.sum(histories[model][0,:]>0))
