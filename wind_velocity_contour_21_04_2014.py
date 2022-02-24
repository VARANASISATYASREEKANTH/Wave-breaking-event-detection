# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 15:01:59 2021

@author: asdg
"""
#import pandas as pd
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
#---------for LSTM-----------
import tensorflow as tf
from numpy import array
from tensorflow.keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
import keras
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import math
from random import *
from math import log 
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from sklearn import preprocessing
from keras.utils.vis_utils import plot_model

plt.rcParams.update({'font.size':22})
params = {'backend': 'ps',
          'axes.labelsize': 22,
          'legend.fontsize': 22,
          'xtick.labelsize': 22,
          'ytick.labelsize': 22,
          'text.usetex': True};

#----vertical wind velocity------
V_21_04_2014=[[1.8,2.5,2.4,1.3],[-3.9,-1,4e-2,-1.3],[-9.9,-4.6,-2.3,-3.8],
   [-11.4,-4.6,-1.7,3.7],[-7.6,-0.7,1.9,-0.3],[-1.7,4.3,6.5,4.7],
   [3.6,8.3,10.7,10.6],[8.7,13.6,17.4,19.3],[14.6,23,29.9,33.4],[20.4,34,44.3,48.1],[23.1,38.5,49,51.6]
   ];


V_07_04_2014=[[-.5,-2.2,-5.1,-8.5],[-3.9,-6.2,-10.9,-15.7],[-4.3,-8.9,-15.3,-20.9],[-0.8,-7.3,14.8,-20.8],
              [6.5,-1.2,-9,-14.5],[13.4,6.5,0.1,-3.9],[17.1,13.9,10.8,21.7],[22.1,22.8,22.3,31.5],
              [34.3,35.5,34,34.5],[49.9,47.9,41.8,34.9],[55.6,49.1,38.3,27]
              ]
    



fig = plt.figure(figsize=(10,8))
plt.figure(1)
ax=plt.subplot(2,1,1)
start, stop, n_values = 0, 4, 4
start1, stop1, n_values1 = 70, 90, 11
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y=np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y,V_21_04_2014 ,cmap='jet')
plt.colorbar(cp)
#plt.clim(0, 0.015)
ax.set_title('' )
ax.set_xlabel('Time(minutes) \n (a)')
ax.set_ylabel('Height(km)')
'''ax.set_xticks(np.arange(0, 360, 90))
ax.set_yticks(np.arange(30, 90, 10))
'''

#plt.xlim(0,160);
plt.ylim(70,90);


ax=plt.subplot(2,1,2)
start, stop, n_values = 0, 4, 4
start1, stop1, n_values1 = 70, 90, 11
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y=np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y,V_07_04_2014 ,cmap='jet')
plt.colorbar(cp)
#plt.clim(0, 0.015)
ax.set_title('' )
ax.set_xlabel('Time(minutes) \n (a)')
ax.set_ylabel('Height(km)')
'''ax.set_xticks(np.arange(0, 360, 90))
ax.set_yticks(np.arange(30, 90, 10))
'''

#plt.xlim(0,160);
plt.ylim(70,90);



