# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:31:54 2021

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
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
import scipy
from sklearn.neighbors import KernelDensity
import numpy as np
from numpy import where, random, array, quantile
from sklearn.neighbors import KernelDensity
import numpy as np
from numpy import where, random, array, quantile



plt.rcParams.update({'font.size':22})
params = {'backend': 'ps',
          'axes.labelsize': 22,
          'legend.fontsize': 22,
          'xtick.labelsize': 22,
          'ytick.labelsize': 22,
          'text.usetex': True};

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

loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/test_data_07_04_2014_ver2.xlsx')
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




###################CLSTM model#########################
seq = Sequential()
seq.add(ConvLSTM2D(filters=1, kernel_size=(3, 3),input_shape=(None, 64, 128, 1),padding='same', return_sequences=True))
seq.add(BatchNormalization());
seq.add(ConvLSTM2D(filters=1, kernel_size=(3, 3),padding='same', return_sequences=True));
seq.add(BatchNormalization());
seq.add(ConvLSTM2D(filters=1, kernel_size=(3, 3),padding='same', return_sequences=True));
seq.add(BatchNormalization());
seq.add(ConvLSTM2D(filters=1, kernel_size=(3, 3),padding='same', return_sequences=True));
seq.add(BatchNormalization());
seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),activation='sigmoid',padding='same', data_format='channels_last'))
#seq.compile(loss='binary_crossentropy', optimizer='adadelta')
seq.compile(optimizer='adam', loss='mse',metrics=['accuracy','mse', 'mae', 'mape'])
seq.summary();

#-----reshaping train and test data------
y=np.zeros(212992);
y_data=y.reshape(1,26, 64, 128, 1);
data =T_pert.reshape(1,26, 64, 128, 1);#training
data_test=test_data_pert.reshape(26, 64, 128, 1);# reshaping test data


#------training the model---
seq.fit(data, y_data, batch_size=10,epochs=20);
#test_scores = seq.evaluate(data_test, y_data, verbose=2);


#----model performance metrics
model_loss=seq.history;
model_metrics=seq.metrics_names;
keras.utils.plot_model(seq, "CLSTM_WAVE_BREAKING_model.pdf");
print(model_metrics);

data_test=test_data_pert.reshape(1,26, 64, 128, 1);
#----testing the model with test data------
out_predicted =seq.predict(data_test);
y_out=out_predicted.reshape(128,1664);
y_out_reduced=np.zeros((128,128));

#-----contour plot-------
fig = plt.figure(figsize=(10,8))
plt.figure(1)

ax=plt.subplot(1,1,1)
start, stop, n_values = 1, 1664, 1664
start1, stop1, n_values1 = 42, 80, 128
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y=np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y,y_out,cmap='hsv')
plt.colorbar(cp)
#plt.clim(0, 0.7)
ax.set_title('' )
ax.set_xlabel('Time(minutes) \n (a)')
ax.set_ylabel('Height(km)')
'''ax.set_xticks(np.arange(0, 360, 90))
ax.set_yticks(np.arange(30, 90, 10))
'''
plt.xlim(0,1664);
plt.ylim(30,80);
########################### performance metrics #####################
#---T2 and SPE metric---
#---SPE metric---
SPE_metric=np.zeros((128,60));
for i in range(0,127,1):
    for j in range(0,59,1):
        SPE_metric[i][j]=np.square(y_out[i][j]-test_data_pert[i][j]);
        #SPE_metric[i][j]=y_out[i][j];


#######################################################################
#################---Kernel Density Estimation(KDE):SPE--- ###########
#----
Z_overall=np.zeros((128,60));
threshold=np.zeros(128);
for t in range(1,6,1):
    #---------computing the overall threshold---
    [q,f,patches] = plt.hist(SPE_metric[t][1:60], 60, facecolor='b',density=True, alpha=1,edgecolor = 'black');
    #---selection of optimal bandwidth---
    opt_mean=np.mean(SPE_metric[t][1:60]);
    opt_std=np.power(np.var(SPE_metric[t][1:60]),0.5);
    s=60;# no  of samples
    H_opt=1.06*opt_std*np.power(s,-0.2);
    Q=np.zeros((60,2));
    #----
    for i in range(0,59,1):
        Q[i][0]=f[i];
        Q[i][1]=q[i];
    kde = KernelDensity(algorithm='auto', bandwidth=H_opt,kernel='gaussian', metric='euclidean',metric_params=None, rtol=0).fit(Q);
    log_density=kde.score_samples(Q);
    Z=np.exp(log_density);
    Z_overall[t][0:60]=np.transpose(Z[1:60][0])
    threshold[t] = quantile(Z, 0.99);
ov_threshold=np.sum(threshold[0:127])/127;
#-------------distribution function-------------
#plt.plot(f[1:60],q[1:60],color='r',linewidth='2');
#plt.hist(SPE_metric[t][1:60], 60, facecolor='b',density=True, alpha=1,edgecolor = 'black');
HEIGHT=np.arange(25.1,80,0.3);
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
data1 =SPE_metric[t][0:60]
density1 = kde.gaussian_kde(data1)
x1 = np.linspace(0,5000,3000)
y1=density1(x1);
fig = plt.figure(figsize=(10,8))
plt.figure(1)
ax=plt.subplot(1,1,1)
plt.plot(x1, np.exp(y1),color='b',linewidth='3');
plt.plot(x1, np.exp(threshold[120])*np.ones(3000),color='r',linewidth='3');
#plt.hist(SPE_metric[t][1:60], 60, facecolor='b',density=True, alpha=1,edgecolor = 'black');
plt.title("Density Plot of the data");
plt.show();
#plt.ylim(1,1.0005);
fig.savefig("density"+np.str(t)+".pdf", bbox_inches='tight');
#---overall threshold---






