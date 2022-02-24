
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import math
from random import *
from math import log 
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

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

loc1=('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/HWM_zonal_wind_19_01_2011.xlsx')
wb = xlrd.open_workbook(loc1)
sheet = wb.sheet_by_index(0)
zonal_u=np.zeros((sheet.nrows,sheet.ncols));
for i in range(sheet.nrows):
    for j in range(sheet.ncols):
        zonal_u[i][j]=sheet.cell_value(i,j)


fig = plt.figure(figsize=(10,8))
plt.figure(1)
ax=plt.subplot(1,1,1)
start, stop, n_values = 18, 23, 7
start1, stop1, n_values1 = 70, 110, 21
x_vals = np.linspace(start, stop, n_values)
y_vals = np.linspace(start1, stop1, n_values1)
X, Y=np.meshgrid(x_vals, y_vals)
cp = plt.contourf(X, Y,zonal_u,cmap='hsv')
plt.colorbar(cp)
#plt.clim(0, 1)
#cp.set_label('P_B')
ax.set_title('Zonal Wind' )
ax.set_xlabel('Time(minutes)')
ax.set_ylabel('Height(km)')
ax.set_xticks(np.arange(18,24,1))
ax.set_yticks(np.arange(70, 115, 5))
#plt.xlim(0,240);
#plt.ylim(40,80);