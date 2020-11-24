#MAHDI ELHOUSNI, WPI 2020

import numpy as np
import cv2
import utils
import time
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.utils import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.applications.densenet import DenseNet121

from nets import *
from utils import *
from tifffile import *

import sys

datasetName=sys.argv[1] #Vaihingen, DFC2018
if(sys.argv[2]=='False'): correction=False
else: correction=True

cropSize=320

predCheckPointPath='./checkpoints/'+datasetName+'/mtl'
corrCheckPointPath='./checkpoints/'+datasetName+'/refinement'

val_rgb, val_dsm, val_sem = collect_tilenames("val",datasetName)

NUM_VAL_IMAGES = len(val_rgb)

print("number of validation samples " + str(NUM_VAL_IMAGES))

backboneNet=DenseNet121(weights='imagenet', include_top=False, input_tensor=Input(shape=(cropSize,cropSize,3)))

net = MTL(backboneNet, datasetName)
net.load_weights(predCheckPointPath)

if(correction):
  autoencoder=Autoencoder()
  autoencoder.load_weights(corrCheckPointPath)

tile_mse   = 0
total_mse  = 0

tile_rmse  = 0
total_rmse = 0

tile_mae   = 0
total_mae  = 0

tile_time  = 0
total_time = 0

tilesLen=len(val_rgb)

for tile in range(tilesLen):

  print(tile+1)

  rgb_data=[]
  dsm_data=[]

  coordinates=[]

  if(datasetName=='Vaihingen'):
    rgb_tile = np.array(Image.open(val_rgb[tile]))/255
    dsm_tile = np.array(Image.open(val_dsm[tile]))/255

  elif(datasetName=='DFC2018'):
    rgb_tile=np.array(Image.open(val_rgb[tile]))/255
    dsm_tile=np.array(Image.open(val_dsm[2*tile]))
    dem_tile=np.array(Image.open(val_dsm[2*tile+1]))
    dsm_tile=correctTile(dsm_tile)
    dem_tile=correctTile(dem_tile)
    dsm_tile=dsm_tile-dem_tile

  for x1, x2, y1, y2 in sliding_window(rgb_tile, step=int(cropSize/6), window_size=(cropSize,cropSize)):
    coordinates.append([y1,y2,x1,x2])
    rgb_data.append(rgb_tile[y1:y2, x1:x2, :])
    dsm_data.append(dsm_tile[y1:y2, x1:x2])

  pred = np.zeros([2,rgb_tile.shape[0],rgb_tile.shape[1]])
  prob_matrix = gaussian_kernel(rgb_tile.shape[0],rgb_tile.shape[1])

  start = time.time()
  for crop in range(len(rgb_data)):
    cropRGB=rgb_data[crop]
    cropDSM=dsm_data[crop]

    y1,y2,x1,x2=coordinates[crop]
    prob_matrix = gaussian_kernel(cropRGB.shape[0], cropRGB.shape[1])

    dsm_output, sem_output, norm_output = net.call(cropRGB[np.newaxis,...], training=False)

    if(correction):
      correctionInput = tf.concat([dsm_output, norm_output, sem_output, cropRGB[np.newaxis,...]], axis=-1)
      noise=autoencoder.call(correctionInput, training=False)
      dsm_output_copy = dsm_output.numpy().squeeze().copy()
      dsm_output = dsm_output-noise

    dsm_output = dsm_output.numpy().squeeze()

    pred[0,y1:y2,x1:x2] += np.multiply(dsm_output, prob_matrix)
    pred[1,y1:y2,x1:x2] += prob_matrix

  end = time.time()
  gaussian = pred[1]
  pred = np.divide(pred[0], gaussian)

  if(datasetName=='DFC2018'): dsm_tile=dsm_tile[0:pred.shape[0],0:pred.shape[1]]

  tile_mse = np.mean((pred-dsm_tile)**2)
  total_mse+= tile_mse
  print("Tile MSE  : " + str(tile_mse))

  tile_mae = np.mean(np.abs(pred-dsm_tile))
  total_mae+= tile_mae
  print("Tile MAE  : " + str(tile_mae))

  tile_rmse = np.sqrt(np.mean((pred-dsm_tile)**2))
  total_rmse+= tile_rmse
  print("Tile RMSE : " + str(tile_rmse))

  tile_time = end - start
  total_time+= tile_time
  print("Tile time : " + str(tile_time))
  
  filename=val_rgb[tile].split('/')[-1].split('.')[0]
  pred = Image.fromarray(pred)
  pred.save('./output/'+datasetName+'/'+filename+'.tif')

print("Final MSE loss  : " + str(total_mse/tilesLen))
print("Final MAE loss  : " + str(total_mae/tilesLen))
print("Final RMSE loss : " + str(total_rmse/tilesLen))












