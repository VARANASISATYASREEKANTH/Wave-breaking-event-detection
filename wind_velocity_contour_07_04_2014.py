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
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.size':22})
params = {'backend': 'ps',
          'axes.labelsize': 22,
          'legend.fontsize': 22,
          'xtick.labelsize': 22,
          'ytick.labelsize': 22,
          'text.usetex': True};

#----Meridonal wind velocity------
loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/mer_07_04_2014_v2.xlsx')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
V_07_04_2014=np.zeros((sheet.nrows,sheet.ncols));
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        V_07_04_2014[i][j]=sheet.cell_value(i,j);
    




loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/mer_21_04_2014_v2.xls')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
V_21_04_2014=np.zeros((sheet.nrows,sheet.ncols));
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        V_21_04_2014[i][j]=sheet.cell_value(i,j);


fig = plt.figure(figsize=(10,8))
plt.figure(1)
ax=plt.subplot(2,1,1)
start, stop, n_values = 0, 32, 33
start1, stop1, n_values1 = 70, 110, 21
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y=np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y,V_07_04_2014 ,cmap='hsv');
plt.colorbar(cp)
#plt.clim(0, 0.015)
ax.set_title('Meridonal velocity(m/s)' )
ax.set_xlabel('Time(Hours) \n (a)')
ax.set_ylabel('Height(km)')
ax.set_xticks(np.arange(0, 40, 8))
ax.set_yticks(np.arange(70, 120, 10))
plt.ylim(70,110);



ax=plt.subplot(2,1,2)
start, stop, n_values = 0, 24, 25
start1, stop1, n_values1 = 70, 110, 21
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y=np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y,V_21_04_2014 ,cmap='hsv');
plt.colorbar(cp)
#plt.clim(0, 0.015)
ax.set_title('Meridonal velocity(m/s)' )
ax.set_xlabel('Time(Hours) \n (a)')
ax.set_ylabel('Height(km)')
ax.set_xticks(np.arange(0, 30, 6))
ax.set_yticks(np.arange(70, 120, 10))
plt.ylim(70,110);



