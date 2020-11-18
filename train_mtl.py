#MAHDI ELHOUSNI, WPI 2020

import numpy as np
import random

from datetime import datetime
from os import path
from skimage import io

import tensorflow as tf
import utils

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

if(datasetName=='DFC2018'):
  label_codes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
  w1=1.0 #sem
  w2=1.0 #norm
  w3=1.0 #dsm

if(datasetName=='Vaihingen'):
  label_codes = [(255,255,255), (0,0,255), (0,255,255), (0,255,0), (255,255,0), (255,0,0)]
  w1=1.0 #sem
  w2=10.0 #norm
  w3=100.0 #dsm

id2code = {k:v for k,v in enumerate(label_codes)}

decay=False
save=True

lr=0.0002
batchSize=4
numEpochs=20
training_samples=10000
val_freq=1000
train_iters=int(training_samples/batchSize)
cropSize=320

predCheckPointPath='./checkpoints/'+datasetName+'/mtl'
corrCheckPointPath='./checkpoints/'+datasetName+'/refinement'

all_rgb, all_dsm, all_sem = collect_tilenames("train", datasetName)
val_rgb, val_dsm, val_sem = collect_tilenames("val", datasetName)

NUM_TRAIN_IMAGES = len(all_rgb)
NUM_VAL_IMAGES = len(val_rgb)

backboneNet=DenseNet121(weights='imagenet', include_top=False, input_tensor=Input(shape=(cropSize,cropSize,3)))

net = MTL(backboneNet, datasetName)

min_loss=1000

for current_epoch in range(1,numEpochs):
  if(decay and current_epoch>1): lr=lr/2
  optimizer = tf.keras.optimizers.Adam(lr=lr,beta_1=0.9)

  print("Current epoch " + str(current_epoch))
  print("Current LR    " + str(lr)) 
  
  error_ave=0.0
  error_L1=0.0
  error_L2=0.0
  error_L3=0.0

  for iters in range(train_iters):

    idx = random.randint(0,len(all_rgb)-1)

    rgb_batch=[]
    dsm_batch=[]
    sem_batch=[]
    norm_batch=[]

    if(datasetName=='Vaihingen'):
      rgb_tile = np.array(Image.open(all_rgb[idx]))/255
      dsm_tile = np.array(Image.open(all_dsm[idx]))/255
      norm_tile=genNormals(dsm_tile)
      sem_tile=np.array(Image.open(all_sem[idx]))

    elif(datasetName=='DFC2018'):
      rgb_tile=np.array(Image.open(all_rgb[idx]))/255
      dsm_tile=np.array(Image.open(all_dsm[2*idx]))
      dem_tile=np.array(Image.open(all_dsm[2*idx+1]))
      dsm_tile=correctTile(dsm_tile)
      dem_tile=correctTile(dem_tile)
      dsm_tile=dsm_tile-dem_tile
      norm_tile=genNormals(dsm_tile)
      sem_tile=np.array(Image.open(all_sem[idx]))

    for i in range(batchSize):
  
      h = rgb_tile.shape[0]
      w = rgb_tile.shape[1]
      r = random.randint(0,h-cropSize)
      c = random.randint(0,w-cropSize)
      rgb = rgb_tile[r:r+cropSize,c:c+cropSize]
      dsm = dsm_tile[r:r+cropSize,c:c+cropSize]
      sem = sem_tile[r:r+cropSize,c:c+cropSize]
      if(datasetName=='DFC2018'): sem = sem[...,np.newaxis]
      norm = norm_tile[r:r+cropSize,c:c+cropSize]
    
      rgb_batch.append(rgb)
      dsm_batch.append(dsm)
      sem_batch.append(rgb_to_onehot(sem, datasetName, id2code))
      norm_batch.append(norm)

    rgb_batch=np.array(rgb_batch)
    dsm_batch=np.array(dsm_batch)[...,np.newaxis]
    sem_batch=np.array(sem_batch)
    norm_batch=np.array(norm_batch)

    from tensorflow.keras.losses import MeanSquaredError
    from tensorflow.keras.losses import CategoricalCrossentropy

    MSE=MeanSquaredError()
    CCE=CategoricalCrossentropy()
    
    with tf.GradientTape() as tape:
      dsm_out, sem_out, norm_out=net.call(rgb_batch, training=True)
      L1=MSE(dsm_batch.squeeze(),tf.squeeze(dsm_out))
      L2=CCE(sem_batch,sem_out)
      L3=MSE(norm_batch,norm_out)
      total_loss=w1*L2+w2*L3+w3*L1

      print(total_loss)
    
    grads = tape.gradient(total_loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
  
    error_ave=error_ave+total_loss.numpy()
    error_L1=error_L1+L1.numpy()
    error_L2=error_L2+L2.numpy()
    error_L3=error_L3+L3.numpy()

    if iters%val_freq==0 and iters>0:

      print(iters)
      print('total loss : ' + str(error_ave/val_freq))
      print('DSM loss   : ' + str(error_L1/val_freq))
      if(sem_flag and not norm_flag): 
        print('SEM loss   : ' + str(error_L2/val_freq))
      if(not sem_flag and norm_flag): 
        print('NORM loss   : ' + str(error_L3/val_freq))
      if(sem_flag and norm_flag): 
        print('SEM loss   : ' + str(error_L2/val_freq))
        print('NORM loss  : ' + str(error_L3/val_freq))

      if(error_L1/val_freq<min_loss and save):
        net.save_weights(predCheckPointPath)
        min_loss=error_L1/val_freq
        print('dsm train checkpoint saved!')

      error_ave=0.0
      error_L1=0.0
      error_L2=0.0
      error_L3=0.0

  error_ave=0.0
  error_L1=0.0
  error_L2=0.0
  error_L3=0.0



   

  






