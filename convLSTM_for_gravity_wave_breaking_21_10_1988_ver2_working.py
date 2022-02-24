# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:57:59 2021

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
#----------------definition of convlstm---------------------
#--------------96% training data, 4% test data----------
#--------------training data perturbations---------------------

loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/training_data/wave_breaking_train_data_ver2.xlsx')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
T_pert=np.zeros((sheet.nrows,sheet.ncols));
T_dash_pert=np.zeros((sheet.nrows,sheet.ncols));
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        T_dash_pert[i][j]=sheet.cell_value(i,j)

for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        T_pert[i][j]=T_dash_pert[i][j]-np.mean(T_dash_pert[i][:]);


#--------------test data perturbations---------------------

loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/test_data_21_10_1998_ver2.xlsx')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
test_data_pert=np.zeros((sheet.nrows,sheet.ncols));
test_data_dash_pert=np.zeros((sheet.nrows,sheet.ncols));
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        test_data_dash_pert[i][j]=sheet.cell_value(i,j)

for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        test_data_pert[i][j]=test_data_dash_pert[i][j]-np.mean(test_data_dash_pert[i][1:60]);




###############################################################
#-----------------training the model(CNN+LSTM)-----------------

#-----reshaping train and test data----
data =T_pert.reshape(1, 128, 64, 26)#training
data_test=test_data_pert.reshape(1, 128, 64, 26)# reshaping test data


#---------------------CNN model----------------------
model=Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(128,64,26),padding='same'))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.01))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.011))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
#model.add(Flatten())
#model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                
model.summary()
#---------------assigning random weights to the model--------
initializer = tf.keras.initializers.RandomUniform(minval=0., maxval=1.)
weights=initializer(shape=(128, 64));

#----------------conforming the model weights----------------
model_weights=model.get_weights()
#-----------------------see the dimensions of the tensor-------

y=np.zeros(1)
#training the model
model.compile(optimizer='adam', loss='mse',metrics=['accuracy','mse', 'mae', 'mape'])
model_hist=model.fit(data,y, epochs=100)


#evaluating model with test data
test_scores = model.evaluate(data, y, verbose=2)

#----model performance metrics
model_loss=model_hist.history;
model_metrics=model.metrics_names;
#keras.utils.plot_model(model, "train_ConvLSTM_WB1.pdf");
keras.utils.plot_model(model, "CNN_model_rayleigh_lidar.pdf", show_shapes=True); 
#keras.utils.plot_model(model, "train_ConvLSTM_WB.eps", show_shapes=True); 
print(model_metrics);

#plt.plot(model_hist.history['loss'])
plt.plot(model_hist.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()


out_predicted = model.predict(data_test);

for i in range(0,1,15):
    y_out_0=np.transpose(out_predicted[0][:][:][0]);
    y_out_1=np.transpose(out_predicted[0][:][:][1]);
    y_out_2=np.transpose(out_predicted[0][:][:][2]);
    y_out_3=np.transpose(out_predicted[0][:][:][3]);
    y_out_4=np.transpose(out_predicted[0][:][:][4]);
    y_out_5=np.transpose(out_predicted[0][:][:][5]);
    y_out_6=np.transpose(out_predicted[0][:][:][6]);
    y_out_7=np.transpose(out_predicted[0][:][:][7]);
    y_out_8=np.transpose(out_predicted[0][:][:][8]);
    y_out_9=np.transpose(out_predicted[0][:][:][9]);
    y_out_10=np.transpose(out_predicted[0][:][:][10]);
    y_out_11=np.transpose(out_predicted[0][:][:][11]);
    y_out_12=np.transpose(out_predicted[0][:][:][12]);
    y_out_13=np.transpose(out_predicted[0][:][:][13]);
    y_out_14=np.transpose(out_predicted[0][:][:][14]);
    y_out_15=np.transpose(out_predicted[0][:][:][15]);

y_out=np.concatenate((y_out_0,y_out_1,y_out_2,y_out_3,y_out_4,y_out_5,y_out_6, y_out_7,y_out_8,y_out_9,y_out_10,y_out_11,y_out_12,y_out_13,y_out_14,y_out_15),axis=1);
y_normalized=preprocessing.normalize(-y_out);
################################  LSTM #####################################################
#------------define input sequence----------------
y_out_lstm=np.zeros((128,128));
out=np.zeros((128,128));
for i in range(1,128,1):
    out[1][:]=y_out[i][:]
        #------------define input sequence----------------
    seq_in=np.abs(out[1][:]);
    #-----reshape input into [samples, timesteps, features]---
    n_in = len(seq_in)
    seq_in = seq_in.reshape((1, n_in, 1))
    #--------prepare output sequence-------------------
    seq_out = seq_in[:, 1:, :]
    n_out = n_in - 1
    #-------------define encoder---------------
    visible=Input(shape=(n_in,1))
    encoder=LSTM(128, activation='relu')(visible)
    #-----------define reconstruct decoder---------
    decoder1 = RepeatVector(n_in)(encoder)
    decoder1 = LSTM(128, activation='relu', return_sequences=True)(decoder1)
    decoder1 = TimeDistributed(Dense(1))(decoder1)
    #-------------define predict decoder----------
    decoder2 = RepeatVector(n_out)(encoder)
    decoder2 = LSTM(128, activation='relu', return_sequences=True)(decoder2)
    decoder2 = TimeDistributed(Dense(1))(decoder2)
    #-----------tie it together-------------
    model_lstm= Model(inputs=visible, outputs=[decoder1, decoder2])
    model_lstm.compile(optimizer='adam', loss='mse',metrics=['accuracy','mse', 'mae', 'mape'])
    plot_model(model_lstm, show_shapes=True, to_file='composite_lstm_autoencoder.png')
    #-------------fit model-----------
    model_hist_lstm=model_lstm.fit(seq_in, [seq_in,seq_out], epochs=10, verbose=0)
    #-----------demonstrate prediction--------
    yhat_lstm= model_lstm.predict(seq_in, verbose=1);
    
    k=np.zeros(128);
    for j in range(0,127,1):
        k[j]=yhat_lstm[0][0][j][0];
        print(k[j])
    y_out_lstm[i][:]=k;
    
keras.utils.plot_model(model_lstm, "LSTM_model_rayleigh_lidar.pdf", show_shapes=True);    
    
#-------------------------------contour plot------------------
fig = plt.figure(figsize=(10,8))
plt.figure(1)
ax=plt.subplot(2,2,1)
start, stop, n_values = 1, 128, 128
start1, stop1, n_values1 = 42, 80, 128
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y=np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y, y_normalized,cmap='jet')
plt.colorbar(cp)
plt.clim(0, 0.015)
ax.set_title('FFT' )
ax.set_xlabel('Time(minutes) \n (a)')
ax.set_ylabel('Height(km)')
'''ax.set_xticks(np.arange(0, 360, 90))
ax.set_yticks(np.arange(30, 90, 10))
'''

#plt.xlim(0,160);
plt.ylim(60,80);

#-------------------------------------