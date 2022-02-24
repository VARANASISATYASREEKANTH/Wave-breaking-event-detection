# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:15:13 2021

@author: asdg
"""
import xlrd
import numpy as np
import matplotlib.pyplot as plt

import math
from random import *
from math import log 
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
params = {'backend': 'ps',
          'axes.labelsize': 20,
          'legend.fontsize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': True};
epochs=np.arange(1,21,1);
tr_acc_08_05_98=[.52,.54,.62,.68,.47,.79,.837,.8765,.909,.9267,.9432,.9465,.9537,.9591,.960,.9607,.9612,.9625,1,1];
tst_acc_08_05_98=[.5077,.556,.6181,.6406,.6617,.6887,.7257,.7611,.87866,.9038,.9168,.9288,.9405,.9501,.9561,.9591,.9609,.9612,1,1];
tr_loss_08_05_98=[.2834,.2675,.2515,.2417,.2330,.2249,.2162,.2063,.1961,.1853,.1738,.1618,.1496,.1374,.1255,.1147,.1049,0.096,.092,.092];
tst_loss_08_05_98=[.29,.267,.2521,.2432,.2358,.227,.219,.2104,.2002,.1891,.1773,.1649,.1523,.1399,.1285,.1181,.1087,.09,.086,.06];

tr_acc_21_10_98=[.737,.8561,.9120,.9615,.9787,.9913,.9952,.9973,.9988,1,1,1,1,1,1,1,1,1,1,1];
tst_loss_21_10_98=[.8699,.3501,.1641,.1018,.0746,.0549,.0418,.0345,.0289,.0241,.0203,.0180,.0162,.0142,.0120,.0099,.0087,.0080,.0073,.0062];
tr_loss_21_10_98=[.9287,.243,.1325,.0757,0.0552,.0431,.0345,.0285,.0238,.0197,.0168,.0144,.0124,.0108,.0098,.0087,.0078,.0068,.0064,.0054];
tst_acc_21_10_98=[.7157,.8362,.91332,.9477,.9660,.9832,.992,.9958,.9961,.9982,.9985,.9997,.9997,.9997,1,1,1,1,1,1];

tr_acc_22_03_2000=[0.7037,.756,.842,.9038,.9471,.9630,.97,.9796,.9859,.9959,.9943,.9952,.9982,.9987,.9994,1,1,1,1,1];
tst_acc_22_03_2000=[0.7100,.8332,.9008,.9468,.9642,.9781,.9910,.9955,.9982,.9997,.9997,1,1,1,1,1,1,1,1,1];
tr_loss_22_03_2000=[1.0031,.6098,.2499,.156,.100,.0738,.0644,.0545,.043,.0330,.0277,.0250,.0225,.0200,.0172,.0147,.0132,.0120,.0106,.0094];
tst_loss_22_03_2000=[1.138,.288,.156,.0996,.0736,.0572,.0419,.0324,.0281,.0240,.0210,.0186,.0162,.0143,.0122,.015,.0093,.0082,.0072,.0066];



fig = plt.figure(figsize=(6,4));
plt.figure(1)


ax = plt.subplot(3,2,1)
ax.plot(epochs,tr_acc_22_03_2000 ,color='r',marker="D",label="Train",linewidth='3.5');
ax.plot(epochs,tst_acc_22_03_2000 ,color='b',label="Test",linewidth='3.5');
ax.legend()
ax = fig.gca()
#ax.set_title('Accuracy');
#ax.set_xticks(np.arange(100, 300, 20))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (a)');
plt.ylabel('Accuracy');
plt.ylim(0.6,1);
plt.xlim(0,20);
plt.show();



ax = plt.subplot(3,2,2)
ax.plot(epochs,tr_loss_22_03_2000 ,color='r',marker="D",label="Train",linewidth='3.5');
ax.plot(epochs,tst_loss_22_03_2000 ,color='b',label="Test",linewidth='3.5');
ax.legend()
ax = fig.gca()
#ax.set_title('Loss');
#ax.set_xticks(np.arange(100, 300, 20))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (b)');
plt.ylabel('Loss');
plt.ylim(0,1);
plt.xlim(0,20);
plt.show();



ax = plt.subplot(3,2,3)
ax.plot(epochs,tr_acc_21_10_98 ,color='r',marker="D",label="Train",linewidth='3.5');
ax.plot(epochs,tst_acc_21_10_98 ,color='b',label="Test",linewidth='3.5');
ax.legend()
ax = fig.gca()
#ax.set_title('Accuracy');
#ax.set_xticks(np.arange(100, 300, 20))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (c)');
plt.ylabel('Accuracy');
plt.ylim(0.6,1);
plt.xlim(0,20);
plt.show();


ax = plt.subplot(3,2,4)
ax.plot(epochs,tr_loss_21_10_98 ,color='r',marker="D",label="Train",linewidth='3.5');
ax.plot(epochs,tst_loss_21_10_98 ,color='b',label="Test",linewidth='3.5');
ax.legend()
ax = fig.gca()
#ax.set_title('Loss');
#ax.set_xticks(np.arange(100, 300, 20))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (d)');
plt.ylabel('Loss');
plt.ylim(0,1);
plt.xlim(0,20);
plt.show();



ax = plt.subplot(3,2,5)
ax.plot(epochs,tr_acc_08_05_98 ,color='r',marker="D",label="Train",linewidth='3.5');
ax.plot(epochs,tst_acc_08_05_98 ,color='b',label="Test",linewidth='3.5');
ax.legend()
ax = fig.gca()
#ax.set_title('Accuracy');
#ax.set_xticks(np.arange(100, 300, 20))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (e)');
plt.ylabel('Accuracy');
plt.ylim(0.2,1);
plt.xlim(0,20);
plt.show();


ax = plt.subplot(3,2,6)
ax.plot(epochs,tr_loss_08_05_98 ,color='r',marker="D",label="Train",linewidth='3.5');
ax.plot(epochs,tst_loss_08_05_98 ,color='b',label="Test",linewidth='3.5');
ax.legend()
ax = fig.gca()
#ax.set_title('Loss');
#ax.set_xticks(np.arange(100, 300, 20))
#ax.set_yticks(np.arange(30, 120, 10))
#ax.grid(True)
plt.xlabel('Epochs  \n (f)');
plt.ylabel('Loss');
plt.ylim(0,1);
plt.xlim(0,20);
plt.show();