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

loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/test_data_07_04_2014_ver2.xlsx')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
test_data_pert=np.zeros((sheet.nrows,sheet.ncols));
test_data_dash_pert=np.zeros((sheet.nrows,sheet.ncols));
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        test_data_dash_pert[i][j]=sheet.cell_value(i,j)

for i in range(sheet.nrows):
    for j in range(1,59,1):
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
test_scores = model.evaluate(data_test, y, verbose=2)

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
    encoder=LSTM(128, activation='linear')(visible)
    FEATURES=LSTM(128, activation='linear')(visible);

    #-----------define reconstruct decoder---------
    decoder1 = RepeatVector(n_in)(encoder)
    decoder1 = LSTM(128, activation='linear', return_sequences=True)(decoder1)
    decoder1 = TimeDistributed(Dense(1))(decoder1)
    #-------------define predict decoder----------
    decoder2 = RepeatVector(n_out)(encoder)
    decoder2 = LSTM(128, activation='linear', return_sequences=True)(decoder2)
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
#inverse=np.linalg.inv(y_out_lstm);
where_are_NaNs = np.isnan(y_out_lstm)
y_out_lstm[where_are_NaNs] =0;

#---eig_values---
#---eigen values
from numpy.linalg import eig  
eig_val,eig_vec=eig(y_out); 
eig_val_abs=np.abs(eig_val); eig_vec_abs=np.abs(eig_vec);
norm_eig_vec=np.zeros((128,128));
norm_eig_vec=(eig_vec_abs)/np.amax(eig_vec_abs);
import numpy as np
from numpy import where, random, array, quantile
threshold_eig=quantile(eig_vec_abs,0.99);
#----------saving the model-----------    
keras.utils.plot_model(model_lstm, "LSTM_model_rayleigh_lidar.pdf", show_shapes=True);    

#-----------contour plot----------
plt.rcParams.update({'font.size':22})
params = {'backend': 'ps',
          'axes.labelsize': 22,
          'legend.fontsize': 22,
          'xtick.labelsize': 22,
          'ytick.labelsize': 22,
          'text.usetex': True};
fig = plt.figure(figsize=(10,8))
plt.figure(1)
ax=plt.subplot(1,1,1)
start, stop, n_values = 0, 256, 128
start1, stop1, n_values1 = 42, 80, 128
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y=np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y,norm_eig_vec,cmap='hsv')
plt.colorbar(cp)
plt.clim(0, 1)
cp.set_label('P_B')
ax.set_title('Probability of Wave Breaking(P_{B})' )
ax.set_xlabel('Time(minutes)')
ax.set_ylabel('Height(km)')
ax.set_xticks(np.arange(0, 272,16))
ax.set_yticks(np.arange(30, 90, 2))
plt.xlim(0,240);
plt.ylim(40,80);
###########performance metrics####################
#---SPE metric---
var_train=np.zeros((128,1));
SPE1=np.zeros((128,128));

A=np.zeros((128,128));
T_square=np.zeros((128,128));
T_square_mid=np.zeros((128,128));
phi=np.zeros((128,128));
phi_inv=np.zeros((128,128));
for i in range(1,128,1):
    var_train[i]=np.var(T_pert[i][1:1664]);
    for j in range(1,128,1):
        SPE1[i][j]=np.power(y_out_lstm[i][j]-y_out[i][j],2)/var_train[i];
#---T2 metric---
A=y_out;#data in feature space
phi=(np.multiply(A,np.transpose(A)))/127;
phi_inv=np.linalg.inv(phi);
T_square_mid=np.multiply(np.transpose(A),phi_inv);
T_square=np.multiply(T_square_mid,A);
###############################################################################
#--- Determining max index values with  99% confidence intervels
from scipy import stats
import numpy as np
mean_spe, sigma_spe = np.mean(SPE1), np.std(SPE1);
conf_int_spe=stats.norm.interval(0.99, loc=mean_spe, scale=sigma_spe);
mean_T_square, sigma_T_square = np.mean(T_square), np.std(T_square);
conf_int_T_square=stats.norm.interval(0.99, loc=mean_T_square, scale=sigma_T_square);
###############################################################################
#------------KDE:SPE-----------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from sklearn.neighbors import KernelDensity
import numpy as np
from numpy import where, random, array 
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
tot_density=np.zeros((128,128));
fig = plt.figure(figsize=(10,8))
plt.figure(4)
for t in range(100,103,1):
    #---------computing the overall threshold---
    [q,f,patches] = plt.hist(SPE1[t][1:128], 128, facecolor='b',density=True, alpha=1,edgecolor = 'black');
    #---selection of optimal bandwidth---
    opt_mean=np.mean(SPE1);
    opt_std=np.std(SPE1);
    s=16384;# no  of samples
    H_opt_spe=1.06*opt_std*np.power(s,-0.2);
    Q=np.zeros((128,2));
    #----
    for i in range(0,128,1):
        Q[i][0]=f[i];
        Q[i][1]=q[i];
    kde = KernelDensity(algorithm='auto', bandwidth=H_opt_spe,kernel='gaussian', metric='euclidean',metric_params=None, rtol=0).fit(Q);
    log_density=kde.score_samples(Q);
    #plt.plot(f[0:128],np.exp(log_density),color='r', linewidth='2');
    plt.hexbin(f[0:128],np.exp(log_density),bins=128,cmap=plt.cm.hsv)
    tot_density[t][0:128]=np.exp(log_density);
 

#------------KDE:T2 metric-----------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kde
from sklearn.neighbors import KernelDensity
import numpy as np
from numpy import where, random, array 
from scipy.signal import savgol_filter
tot_density1=np.zeros((128,128));
fig = plt.figure(figsize=(10,8))
plt.figure(4)
for t in range(1,128,1):
    #---------computing the overall threshold---
    [q,f,patches] = plt.hist(T_square[t][1:128], 128, facecolor='m',density=True, alpha=1,edgecolor = 'black');
    #---selection of optimal bandwidth---
    opt_mean=np.mean(T_square);
    opt_std=np.std(T_square);
    s=16384;# no  of samples
    H_opt_Tsquare=1.06*opt_std*np.power(s,-0.2);
    Q=np.zeros((128,2));
    #----
    for i in range(0,128,1):
        Q[i][0]=f[i];
        Q[i][1]=q[i];
    kde = KernelDensity(algorithm='auto', bandwidth=H_opt_Tsquare,kernel='gaussian', metric='euclidean',metric_params=None, rtol=0).fit(Q);
    log_density1=kde.score_samples(Q);
    plt.plot(f[0:128],savgol_filter(log_density1,5,3),color='b', linewidth='2');
    tot_density1[t][0:128]=np.exp(log_density1);
 
'''
#---breaking height and time and height---

t_wb_spe=np.zeros(128);
for i in range(1,128,1):
    for j in range(1,128,1):
        if SPE1[i][j]<conf_int_spe[1]:
            SPE1[i][j]=0;
        if T_square[i][j]<conf_int_T_square[1]:
            T_square[i][j]=0;
            
'''

spe_metric=np.zeros((128,128));
spe_metric=SPE1
#---contour plot T2---
fig = plt.figure(figsize=(10,8))
plt.figure(2)
ax=plt.subplot(1,1,1)
start, stop, n_values = 0, 256, 128
start1, stop1, n_values1 = 42, 80, 128
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y=np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y,spe_metric,cmap='jet')
plt.colorbar(cp)
#plt.clim(0, 1e-11)
ax.set_title('21/04/2014' )
ax.set_xlabel('Time(minutes) \n (a)')
ax.set_ylabel('Height(km)')
ax.set_xticks(np.arange(0, 272,16))
ax.set_yticks(np.arange(30, 90, 2))
plt.xlim(0,240);
plt.ylim(40,80);
##########################################
#---plot of 1D KDE---
import pandas as pd
plt.rcParams.update({'font.size':22})
params = {'backend': 'ps',
          'axes.labelsize': 22,
          'legend.fontsize': 22,
          'xtick.labelsize': 22,
          'ytick.labelsize': 22,
          'text.usetex': True};

A=110;
fig = plt.figure(figsize=(10,8))
plt.figure(2)
ax=plt.subplot(1,1,1)
s = pd.Series(spe_metric[A][1:128]);
ax = s.plot.kde(bw_method=0.1,color='b',linewidth=2);
ax.set_title('21/04/2014' );
ax.set_xlabel('SPE');
ax.set_ylabel('Density');
fig = plt.figure(figsize=(30,30))
plt.figure(3)
ax=plt.subplot(1,1,1)
s = pd.Series(T_square[A][1:128]);
ax = s.plot.kde(bw_method=.03,color='k',linewidth=2);
#[p,q]=s.plot.kde(bw_method=.03,color='k',linewidth=2);
plt.grid('True')
ax.set_title('21/04/2014' );
ax.set_xlabel('T_square');
ax.set_ylabel('Density');
spe_mean=np.zeros(128);spe_std=np.zeros(128);spe_var=np.zeros(128);t2_mean=np.zeros(128);t2_var=np.zeros(128);t2_std=np.zeros(128);
for i in range(1,128,1):
    spe_mean[i]=np.mean(spe_metric[i][1:128]);spe_std[i]=np.std(spe_metric[i][1:128]);spe_var[i]=np.var(spe_metric[i][1:128]);
    t2_mean[i]=np.mean(T_square[i][1:128]);spe_std[i]=np.std(T_square[i][1:128]);spe_var[i]=np.var(T_square[i][1:128]);
    
###############################################################################
#######--deriving probability from density---########
####################---using simpson 1/3 rd rule---
#---Input section---
bounds=np.linspace(conf_int_spe[0],conf_int_spe[1],128);
result=np.zeros(128);
for i in range(0,127,1):
    def f(x):
        return (1/np.sqrt(2*3.14*spe_var[110]))*np.exp((-(x-spe_mean[110])*(x-spe_mean[110]))/(2*spe_var[110]));
    # Implementing Simpson's 1/3 
    def simpson13(x0,xn,n):
        #calculating step size
        h = (xn - x0) / n
        # Finding sum 
        integration=f(x0) + f(xn);
        for i in range(1,n):
            k=x0 + i*h;
            if i%2 == 0:
                integration=integration + 2 * f(k);
            else:
                integration=integration + 4 * f(k);
        # Finding final integration value
        integration=integration * h/3;
        return integration
    '''lower_limit = float(input("Enter lower limit of integration: "));
    upper_limit = float(input("Enter upper limit of integration: "));
    sub_interval = int(input("Enter number of sub intervals: "));'''
    # Call trapezoidal() method and get result
    result[i] = simpson13(0, bounds[i], 5);
    print("Integration result by Simpson's 1/3 method is: %0.6f");
#-----------------------------------------------------------------------
result=result/np.amax(result);






