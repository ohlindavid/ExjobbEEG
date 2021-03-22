import os,sys,logging,datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow
tensorflow.get_logger().setLevel(logging.ERROR)
from define_model import define_Morlet, define_2DR, define_RecR, define_1DCNN
from train_modelTransfer import train_model as train_modelTransfer
from train_model import train_model
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

#Directory: C:/Users/david/Documents/GitHub/exjobb/Testing Sets/final/subj(N)/(modellnamn)/(A/B/C)(Trialnummer)
n_subj = 6
models = {
#Tuple: (function,epochs,L,Fs,nchan)
#	"1DCNN":(define_1DCNN,1,769,512,31,"1DCNN"),
	"Morlet":(define_Morlet,50,769,512,31,"Morlet"),
#	"Reassignment2D":(define_Morlet,3,193,128,31,"Reassignment2D"),
#	"ReassignmentRec":(define_RecR,1,193,128,31,"ReassignmentRec")
}

histories = {
#	"1DCNN":np.zeros(models["1DCNN"][1]),
	"Morlet":np.zeros([models["Morlet"][1],n_subj]),
	"MorletPre":np.zeros([models["Morlet"][1],n_subj]),
#	"Reassignment2D":np.zeros(models["Reassignment2D"][1]),
#	"ReassignmentRec":np.zeros(models["ReassignmentRec"][1])
}

for model in models:
	subjects = []
	for folder in os.listdir("C:/Users/Kioskar/Desktop/Testing exjobb/Albin_Damir/AD_crop/" + model + "/"):
		subjects.append(folder[4:6])
	subjects = ["01","02","03","04","05","06"]
	print(subjects)
	print("Model: " + model + ", Epochs: " + str(models[model][1]))
	print("--------------------------------------------------------------------------")
	i = 0
	for subject in subjects:
		print("Subject: " + subject)
		date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		train_modelTransfer(models[model],subject,date)
		vals_pretrain = np.array(train_model(models[model],subject,date,pretrain = True))
		vals = np.array(train_model(models[model],subject,date))
		histories[model][:,i] = np.mean(vals,axis=0)
		histories["MorletPre"][:,i] = np.mean(vals_pretrain,axis=0)
		print(histories[model])
		print(histories["MorletPre"])
		i = i + 1
		print(np.sum(histories[model],axis=1)/np.sum(histories[model][0,:]>0))
		print(np.sum(histories["MorletPre"],axis=1)/np.sum(histories["MorletPre"][0,:]>0))


	#histories[model] = histories[model]/len(subjects)
