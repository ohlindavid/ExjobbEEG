import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers, losses, Input, activations, utils
from MorletLayer import MorletConv
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os,sys
import tensorflow.keras.backend as K
from settings import who, epochs, checkpoint_path
from generator import signalLoader
import math
import datetime
from IPython.display import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from split_data import split_data

def make_gradcam_heatmap(
    img_array, model, last_conv_layer_name, classifier_layer_names
):
    real_pred = np.where(img_array[1] == 1)[1][0]
    imag_array = img_array[0]

    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    heatmap_classes = np.zeros([1958,3])
    with tf.GradientTape() as tape:
    # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(imag_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, real_pred]
        # This is the gradient of the top predicted class with regard to
        # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0,2,3))
    print(pooled_grads)
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[i, :, 0] = pooled_grads[i]*last_conv_layer_output[i, :, 0]
            # The channel-wise mean of the resulting feature map
            # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=(0,2))
            # For visualization purpose, we will also normalize the heatmap between 0 & 1
    print(heatmap)
    print(np.max(heatmap,0))
    heatmap = np.maximum(heatmap,0)
        #heatmap = heatmap - np.minimum(heatmap)
    #if np.max(heatmap) > 0:
    heatmap = heatmap / np.max(heatmap)
    heatmap_classes[:,real_pred] = heatmap
    print(heatmap_classes)
    return heatmap_classes


def generate_gradCAM(model,spex,path,inputVal,vallen):
    (_,_,L,Fs,nchan,modelName) = spex
    names = os.listdir(path)
    np.random.shuffle(names)

    # Make model
    last_conv_layer_name = "second_permute"
    classifier_layer_names = ["pooling","dropout","flatten","dense"]
    #(inputVal,_,vallen,_) = split_data(["A","B","C"],100,0,names,path,spex,batch_size=1)

    heatmap_mean= np.zeros([vallen,1958,3])
    heatmap = heatmap_mean
    for i in range(0,vallen):
        im = next(inputVal)
        guess = model.predict(x=im[0])
        #print(np.argmax(guess))
        #print(np.argmax(im[1]))
        # Generate class activation heatmap
        if np.argmax(guess)==np.argmax(im[1]):
            heatmap[i,:,:] = make_gradcam_heatmap(im, model, last_conv_layer_name, classifier_layer_names)
    #print(np.shape(heatmap_mean))
    heatmap_mean = np.sum(heatmap,axis=0)/(np.sum(np.abs(heatmap),axis=(0,1))+1*(np.sum(heatmap>0,axis=(0,1))==0))
    #print(np.shape(heatmap_mean))
    return heatmap_mean.T
