# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 09:49:42 2021

@author: asdg
"""
#---for convolution layer-----
import keras
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






plt.rcParams.update({'font.size':28})
params = {'backend': 'ps',
          'axes.labelsize': 26,
          'legend.fontsize': 26,
          'xtick.labelsize': 26,
          'ytick.labelsize': 26,
          'text.usetex': True};
#--------------80% train data, 20% test data----------
T_pert=np.zeros((184,90));
T_dash_pert=np.zeros((184,90));
loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/temperature_perturbations_hc_22_mar_2000.xls')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        T_pert[i][j]=sheet.cell_value(i,j);
    
        #if ((T_pert[i][j]>400) or (T_pert[i][j]<-400)):
            
            #T_pert[i][j]=40;
            
            
            
#-----------pre processing------


'''for i in range(1,184,1):
    for j in range(1,60,1):
        T_pert[i][j]=T_dash_pert[i][j]
        if T_pert[i][j]>200:
            T_pert[i][j]=200;
        if T_pert[i][j]<-200:
            T_pert[i][j]=-200;'''

#-------performing the FFT Transform-----------------   

     
T_per_fft=(np.fft.fft2(T_pert, s=None, axes=(- 2, - 1), norm=None)); 
T_per_fft_abs=np.abs(np.fft.fft2(T_pert, s=None, axes=(- 2, - 1), norm=None));
#T_filtered=butter_bandpass_filter(T_per_fft_abs, 0.2, 0.4, 1, order=5)


#-------------------------butter worth bandpass filter-------------
from scipy import signal
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=8):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=8):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
T_filtered=butter_bandpass_filter(T_per_fft_abs, 2, 2.5,10, order=8)
T_filtered=butter_bandpass_filter(np.transpose(T_filtered), 1.8, 2,10, order=8)
T_filtered=np.transpose(T_filtered);
#sos = signal.butter(0.2, 0.6, 'hp', fs=1, output='sos') #Sampling rate is 1000hz, bandwidth is 15hz, output sos
#filtered = signal.sosfilt(sos,T_per_fft_abs ) #The signal is passed through the filter to get the filtered result. Here sos is a bit like a shock response, this function is a bit like a convolution.
#T_per_fft_abs=T_per_fft_abs.butter(2, 4, 'hp', fs=1, output='sos')

############################################################## 
#-----------Convolution Neural Network-------------
from numpy import asarray
from keras.models import Sequential
from keras.layers import Conv2D

#define input data
data1 = [[2, 3, 4, 1, 1, 0, 0, 0],
		[0, 3, 6, 1, 1, 8, 3, 2],
		[3, 5, 7, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0],
		[0, 0, 0, 1, 1, 0, 0, 0]]
data1 = np.fft.fft(T_filtered);

data = data1.reshape(1, 184, 90, 1)#training

#--------create model
model = Sequential()
model.add(Conv2D(1, (3,3), input_shape=(4, 4, 1)))
#---adding several layers----

'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(8,8,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='softmax'))
'''




#define a vertical line detector
detector = [[[[0]],[[1]],[[0]]],
            [[[1]],[[2]],[[1]]],
            [[[1]],[[3]],[[3]]]]
weights = [asarray(detector), asarray([0.0])]

#store the weights in the model
model.set_weights(weights)


#confirm they were stored
model_weights=model.get_weights()


out=np.zeros((182,90));
#apply filter to input data
yhat = model.predict(data)#provide the testing data
for j in range(1,182,1):
    for k in range(1,58,1):
        out[j][k]=yhat[0][j][k][0];
'''for r in range(yhat.shape[1]):
	# print each column in the row
	out=yhat[0][r][:][0]; '''
    
print(yhat);





##################################################
#-------------------------------------------------
#------------------LSTM network-------------------
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
encoder=LSTM(100, activation='relu')(visible)
#-----------define reconstruct decoder---------
decoder1 = RepeatVector(n_in)(encoder)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)
#-------------define predict decoder----------
decoder2 = RepeatVector(n_out)(encoder)
decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)
#-----------tie it together-------------
model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='adam', loss='mse',metrics=['accuracy','mse', 'mae', 'mape'])
plot_model(model, show_shapes=True, to_file='composite_lstm_autoencoder.png')
#-------------fit model-----------
model_hist=model.fit(seq_in, [seq_in,seq_out], epochs=1000, verbose=0)
#-----------demonstrate prediction--------
yhat_lstm_out=np.zeros(( 182, 88));
for i in range(1,182,1):
    seq_in=out[i][:];
    print(seq_in);
    yhat = model.predict(out[i][:], verbose=1)
    for j in range(1,88,1):
        yhat_lstm_out[i][j]=np.transpose(yhat[0][j][0]);










#-----calculation of model validation and accuracy----------
print('')
model_metrics=model.metrics_names
model_loss=model_hist.history;
#out=np.fft.ifft(yhat);




#--------------contour plot-----------------
#--------------before LSTM------------------







fig = plt.figure(figsize=(5,20))
plt.figure(1)
ax=plt.subplot(2,2,1)
start, stop, n_values = 4, 596, 90
start1, stop1, n_values1 = 25, 80, 184
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y = np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y, T_per_fft_abs, cmap='hsv')
plt.colorbar(cp)
ax.set_title('FFT' )
ax.set_xlabel('Time(minutes) \n (a)')
ax.set_ylabel('Height(km)')
ax.set_xticks(np.arange(0, 600, 60))
ax.set_yticks(np.arange(30, 90, 10))
plt.xlim(0,160);
plt.ylim(30,80);




'''
ax=plt.subplot(2,2,2)
start, stop, n_values = 4, 240, 58
start1, stop1, n_values1 = 25, 80, 182
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y = np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y, out,cmap='jet')
plt.colorbar(cp)
ax.set_title('CNN' )
ax.set_xlabel('Time(minutes)  \n (b)')
ax.set_ylabel('Height(km)')
plt.xlim(0,160);
plt.ylim(30,80);




#---------------------

#----get data for only LSTM



ax=plt.subplot(2,2,3)
start, stop, n_values = 4, 240, 61
start1, stop1, n_values1 = 25, 80, 184
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y = np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y, T_pert,cmap='jet')
plt.colorbar(cp)
ax.set_title('LSTM' )
ax.set_xlabel('Time(minutes) \n (c)')
ax.set_ylabel('Height(km)')
plt.xlim(0,160);
plt.ylim(30,80);


ax=plt.subplot(2,2,4)
start, stop, n_values = 4, 240, 58
start1, stop1, n_values1 = 25, 80, 182
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y = np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y, yhat_lstm_out,cmap='jet')
plt.colorbar(cp)
ax.set_title('CNN+LSTM' )
ax.set_xlabel('Time(minutes) \n (d)')
ax.set_ylabel('Height(km)')
plt.clim(-25,25)
plt.xlim(0,160);
plt.ylim(30,80);


#plt.savefig('only_fft_LSTM_CNN.pdf',dpi=800);
plt.show()

'''
#---------------------------------------------------------
#----------calculation of evaluation metrics--------------
#---------------------------------------------------------
#k[:][0]=model_loss[0][:][0]
fig = plt.figure(figsize=(12,5));
plt.figure(2)
e_pochs=np.arange(0,100,1);
ax = plt.subplot(3,2,1)
#ax.plot(e_pochs, np.transpose(model_loss[0][:][0]),color='C3',label="Loss",linewidth='2.5');
#ax.plot(savgol_filter(temp_kktl2_28_dec_98[100:275],11,1),height_inversion[100:275],color='K',label="$DL+KKTl_2$",linewidth='2.5');
ax.legend()
ax = fig.gca()
ax.set_title('Loss')
#ax.set_xticks(np.arange(100, 300, 20))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (a)');
plt.ylabel('validation loss');
#plt.xticks(np.linspace(150,300,6));
#plt.ylim(30,80)



ax = plt.subplot(3,2,2)
#ax.plot(savgol_filter(temp_hc_28_dec_98[100:270],11,1),height_inversion[100:270],color='C3',label="HC",linewidth='2.5');
#ax.plot(savgol_filter(temp_kktl2_28_dec_98[100:275],11,1),height_inversion[100:275],color='K',label="$DL+KKTl_2$",linewidth='2.5');
ax.legend()
ax = fig.gca()
ax.set_title('Evaluation Metrics')
#ax.set_xticks(np.arange(100, 300, 20))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (b)');
plt.ylabel('Accuracy');
#plt.xticks(np.linspace(150,300,6));
#plt.ylim(30,80)



ax = plt.subplot(3,2,3)
#ax.plot(savgol_filter(temp_hc_28_dec_98[100:270],11,1),height_inversion[100:270],color='C3',label="HC",linewidth='2.5');
#ax.plot(savgol_filter(temp_kktl2_28_dec_98[100:275],11,1),height_inversion[100:275],color='K',label="$DL+KKTl_2$",linewidth='2.5');
ax.legend()
ax = fig.gca()
ax.set_title('Receiver Operating Characteristic(ROC)')
#ax.set_xticks(np.arange(100, 300, 20))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (b)');
plt.ylabel('Accuracy');
#plt.xticks(np.linspace(150,300,6));
#plt.ylim(30,80)




ax = plt.subplot(3,2,4)
#ax.plot(savgol_filter(temp_hc_28_dec_98[100:270],11,1),height_inversion[100:270],color='C3',label="HC",linewidth='2.5');
#ax.plot(savgol_filter(temp_kktl2_28_dec_98[100:275],11,1),height_inversion[100:275],color='K',label="$DL+KKTl_2$",linewidth='2.5');
ax.legend()
ax = fig.gca()
ax.set_title('F_1-Score')
#ax.set_xticks(np.arange(100, 300, 20))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (b)');
plt.ylabel('Accuracy');
#plt.xticks(np.linspace(150,300,6));
#plt.ylim(30,80)







ax = plt.subplot(3,2,5)
#ax.plot(savgol_filter(temp_hc_28_dec_98[100:270],11,1),height_inversion[100:270],color='C3',label="HC",linewidth='2.5');
#ax.plot(savgol_filter(temp_kktl2_28_dec_98[100:275],11,1),height_inversion[100:275],color='K',label="$DL+KKTl_2$",linewidth='2.5');
ax.legend()
ax = fig.gca()
ax.set_title('Recall')
#ax.set_xticks(np.arange(100, 300, 20))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (b)');
plt.ylabel('Accuracy');
#plt.xticks(np.linspace(150,300,6));
#plt.ylim(30,80)

ax = plt.subplot(3,2,6)
#ax.plot(savgol_filter(temp_hc_28_dec_98[100:270],11,1),height_inversion[100:270],color='C3',label="HC",linewidth='2.5');
#ax.plot(savgol_filter(temp_kktl2_28_dec_98[100:275],11,1),height_inversion[100:275],color='K',label="$DL+KKTl_2$",linewidth='2.5');
ax.legend()
ax = fig.gca()
ax.set_title('Precision')
#ax.set_xticks(np.arange(100, 300, 20))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (b)');
plt.ylabel('Accuracy');
#plt.xticks(np.linspace(150,300,6));
#plt.ylim(30,80)


#---------------Plot of Confusion Matrix--------
'''
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


cm = confusion_matrix(data1[1:182][1:38], yhat_lstm_out[1:182][1:38])
cm_display = ConfusionMatrixDisplay(cm).plot()'''