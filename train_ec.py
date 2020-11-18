#MAHDI ELHOUSNI, WPI 2020

import numpy as np
import random
import utils

import tensorflow as tf

from skimage import io
from os import path

from tensorflow.keras import backend as K
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.applications.densenet import DenseNet121

from nets import *
from utils import *

import sys

datasetName=sys.argv[1] #Vaihingen, DFC2018

predCheckPointPath='./checkpoints/'+datasetName+'/mtl'
corrCheckPointPath='./checkpoints/'+datasetName+'/refinement'

save=True

lr=0.0002
batchSize=2
numEpochs=20
training_samples=10000
val_freq=1000
cropSize=320

train_iters=int(training_samples/batchSize)

error_ave=0.0

all_rgb, all_dsm, all_sem = collect_tilenames("train", datasetName)
val_rgb, val_dsm, val_sem = collect_tilenames("val", datasetName)

print(all_rgb)
print(val_rgb)

NUM_TRAIN_IMAGES = len(all_rgb)
NUM_VAL_IMAGES = len(val_rgb)

print("number of training samples " + str(NUM_TRAIN_IMAGES))
print("number of validation samples " + str(NUM_VAL_IMAGES))

backboneNet=DenseNet121(weights='imagenet', include_top=False, input_tensor=Input(shape=(cropSize,cropSize,3)))

net=MTL(backboneNet, datasetName)
net.load_weights(predCheckPointPath)

autoencoder=Autoencoder()
optimizer=tf.keras.optimizers.Adam(lr=lr,beta_1=0.9)

min_loss=1000

for epoch in range(1,numEpochs):

  print('Current epoch: ' + str(epoch))

  for iters in range(train_iters):

    idx = random.randint(0,len(all_rgb)-1)

    rgb_batch=[]
    dsm_batch=[]

    if(datasetName=='Vaihingen'):
      rgb_tile = np.array(Image.open(all_rgb[idx]))/255
      dsm_tile = np.array(Image.open(all_dsm[idx]))/255

    elif(datasetName=='DFC2018'):
      rgb_tile=np.array(Image.open(all_rgb[idx]))/255
      dsm_tile=np.array(Image.open(all_dsm[2*idx]))
      dem_tile=np.array(Image.open(all_dsm[2*idx+1]))
      dsm_tile=correctTile(dsm_tile)
      dem_tile=correctTile(dem_tile)
      dsm_tile=dsm_tile-dem_tile
  
    for i in range(batchSize):
  
      h = rgb_tile.shape[0]
      w = rgb_tile.shape[1]
      r = random.randint(0,h-cropSize)
      c = random.randint(0,w-cropSize)
      rgb = (rgb_tile[r:r+cropSize,c:c+cropSize])
      dsm = (dsm_tile[r:r+cropSize,c:c+cropSize])[...,np.newaxis]

      rgb_batch.append(rgb)
      dsm_batch.append(dsm)

    rgb_batch=np.array(rgb_batch)
    dsm_batch=np.array(dsm_batch)

    dsm_out, sem_out, norm_out=net.call(rgb_batch, training=False)
    correctionInput = tf.concat([dsm_out, norm_out, sem_out, rgb_batch], axis=-1)

    from tensorflow.keras.losses import MeanSquaredError
    MSE=MeanSquaredError()
    
    with tf.GradientTape() as tape:
      noise=autoencoder.call(correctionInput, training=True)
      dsm_out=dsm_out-noise
      total_loss=MSE(dsm_batch,dsm_out)
    
    grads = tape.gradient(total_loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables))
  
    error_ave=error_ave+total_loss.numpy()

    if iters%val_freq==0 and iters>0:

      print(iters)
      print('total loss : ' + str(error_ave/val_freq))

      if(error_ave/val_freq<min_loss and save): 
        autoencoder.save_weights(corrCheckPointPath)
        min_loss=error_ave/val_freq
        print('train checkpoint saved!')

      error_ave=0.0

  error_ave=0.0


        
















