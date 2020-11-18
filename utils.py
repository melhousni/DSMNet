#MAHDI ELHOUSNI, WPI 2020

import numpy as np
import glob
import cv2
import PIL

import tensorflow as tf

from os import path
from PIL import Image
from skimage import io

Image.MAX_IMAGE_PIXELS = 1000000000

def collect_tilenames(mode, dataset):

  all_rgb = []
  all_dsm = []
  all_sem = []

  if(dataset=='Vaihingen'):
    trainFrames=[1,3,5,7,11,13,15,17,21,23,26,28,30,32,34,37]
    valFrames=[2,4,6,8,10,12,14,16,20,22,24,27,29,31,33,35,38]
  elif(dataset=='DFC2018'):
    trainFrames = ['UH_NAD83_272056_3289689', 'UH_NAD83_272652_3289689', 'UH_NAD83_273248_3289689', 'UH_NAD83_273844_3289689']
    valFrames = ['UH_NAD83_271460_3289689', 'UH_NAD83_271460_3290290', 'UH_NAD83_272056_3290290', 'UH_NAD83_272652_3290290', 'UH_NAD83_273248_3290290', 'UH_NAD83_273844_3290290', 'UH_NAD83_274440_3289689', 'UH_NAD83_274440_3290290', 'UH_NAD83_275036_3289689', 'UH_NAD83_275036_3290290']

  if(mode=='train'):
    for i in trainFrames:
      if(dataset=='Vaihingen'):
        all_rgb.append('./datasets/Vaihingen/RGB/top_mosaic_09cm_area'+str(i)+'.tif')
        all_dsm.append('./datasets/Vaihingen/NDSM/dsm_09cm_matching_area'+str(i)+'.jpg')
        all_sem.append('./datasets/Vaihingen/SEM/top_mosaic_09cm_area'+str(i)+'.tif')
      elif(dataset=='DFC2018'):
        all_rgb.append('./datasets/DFC2018/RGB/'+i+'.tif')
        all_dsm.append('./datasets/DFC2018/DSM/'+i+'.tif')
        all_dsm.append('./datasets/DFC2018/DEM/'+i+'.tif') 
        all_sem.append('./datasets/DFC2018/SEM/'+i+'.tif')
  elif(mode=='val'):
    for i in valFrames:
      if(dataset=='Vaihingen'):
        all_rgb.append('./datasets/Vaihingen/RGB/top_mosaic_09cm_area'+str(i)+'.tif')
        all_dsm.append('./datasets/Vaihingen/NDSM/dsm_09cm_matching_area'+str(i)+'.jpg')
        all_sem.append('./datasets/Vaihingen/SEM/top_mosaic_09cm_area'+str(i)+'.tif')
      elif(dataset=='DFC2018'):
        all_rgb.append('./datasets/DFC2018/RGB/'+i+'.tif')
        all_dsm.append('./datasets/DFC2018/DSM/'+i+'.tif')
        all_dsm.append('./datasets/DFC2018/DEM/'+i+'.tif') 
        all_sem.append('./datasets/DFC2018/SEM/'+i+'.tif')

  return all_rgb, all_dsm, all_sem

def rgb_to_onehot(rgb_image, dataset, colormap):
    num_classes = len(colormap)
    shape = rgb_image.shape[:2]+(num_classes,)
    encoded_image = np.zeros( shape, dtype=np.int8 )
    for i, cls in enumerate(colormap):
      if(dataset=='DFC2018'): encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,1) ) == colormap[i], axis=1).reshape(shape[:2])
      elif(dataset=='Vaihingen'): encoded_image[:,:,i] = np.all(rgb_image.reshape( (-1,3) ) == colormap[i], axis=1).reshape(shape[:2])
    return encoded_image

def genNormals(dsm_tile, mode='sobel'):

  if(mode=='gradient'):
    zy, zx = np.gradient(dsm_tile) 
  elif(mode=='sobel'):
    zx = cv2.Sobel(dsm_tile, cv2.CV_64F, 1, 0, ksize=5)     
    zy = cv2.Sobel(dsm_tile, cv2.CV_64F, 0, 1, ksize=5)
 
  norm_tile = np.dstack((-zx, -zy, np.ones_like(dsm_tile)))
  n = np.linalg.norm(norm_tile, axis=2)
  norm_tile[:, :, 0] /= n
  norm_tile[:, :, 1] /= n
  norm_tile[:, :, 2] /= n
  
  norm_tile += 1
  norm_tile /= 2
  
  return norm_tile

def correctTile(tile):

  tile[tile > 1000] = -123456
  tile[tile == -123456] = np.max(tile)
  tile[tile < -1000] = 123456
  tile[tile == 123456] = np.min(tile)

  return tile

def gaussian_kernel(width, height, sigma=0.2, mu=0.0):
  x, y = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, height))
  d = np.sqrt(x*x+y*y)
  gaussian_k = (np.exp(-((d-mu)**2 / (2.0 * sigma**2)))) / np.sqrt(2 * np.pi * sigma**2)
  return gaussian_k # / gaussian_k.sum()

def sliding_window(image, step, window_size):
  height, width = (image.shape[0], image.shape[1])
  for x in range(0, width, step):
    if x + window_size[0] >= width:
      x = width - window_size[0]
    for y in range(0, height, step):
      if y + window_size[1] >= height:
        y = height - window_size[1]
      yield x, x + window_size[0], y, y + window_size[1]



