# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:10:26 2019

@author: asdg
"""
#coding:latin_1
import cv2
from Functions import *
from PIL import Image
from scipy.misc import imsave
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import xlrd
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 1. Initial settings. -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
resize_shape = (8,8)  # Resized image's shape
sigma=5               # Noise standard dev.
window_shape = (1, 168) # Patches' shape
step = 21                # Patches' step
ratio =3.35# Ratio for the dictionary (training set).
ksvd_iter =5              # Number of iterations for the K-SVD.
#-------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------- 2. data import ----------------------------------------------#
#-----------------------------------------------photon count data--------------------------------------------------------------------#

loc_d = ('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3_study_of_gravity_waves_with_rayleigh_lidar/working_programs/programs/ksvd_GW_analysis/temperature_perturbations_hc_22_mar_2000.xlsx')

wb = xlrd.open_workbook(loc_d)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
learning_data=np.zeros((sheet.nrows,sheet.ncols));  
for k in range(sheet.nrows):
    for m in range(sheet.ncols):
        learning_data[k][m]=sheet.cell_value(k,m);


loc_d = ('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3_study_of_gravity_waves_with_rayleigh_lidar/working_programs/programs/ksvd_GW_analysis/temperature_perturbations_hc_22_mar_2000.xlsx')
wb = xlrd.open_workbook(loc_d)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
testing_data=np.zeros((sheet.nrows,sheet.ncols)); 
for k in range(sheet.nrows):
    for m in range(sheet.ncols):
        testing_data[k][m]=sheet.cell_value(k,m);


'''img = cv2.imread('G:/research_works_vssreekanth_jrf/MY_PAPERS/sparsity_including_norms_on_frame_expansion/programs/compressive_sensing_sparsity/K-SVD-master___working_code_for_updation/raw_photon_count.png')
# Here I want to convert img in 32 bits
cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB, img)
# Some image processing ...
cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR, img)
cv2.imwrite('G:/research_works_vssreekanth_jrf/MY_PAPERS/sparsity_including_norms_on_frame_expansion/programs/compressive_sensing_sparsity/K-SVD-master___working_code_for_updation/raw_photon_count/out_32', img, [cv2.cv.CV_IMWRITE_PNG_COMPRESSION, 0])

'''
#######################################ksvd  performance with test signals#############
 



#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- 3. Image processing. --------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- 4. Denoising. -----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

denoised_data, calc_time, n_total = denoising(testing_data, learning_data, window_shape, step, sigma, ratio, ksvd_iter)

#######################plot ensemble of  test data and denoised data##################


loc_d = ('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3_study_of_gravity_waves_with_rayleigh_lidar/working_programs/programs/ksvd_GW_analysis/dictionary_out_temp_perturbations_22_mar_2000_index_removed.xlsx')
wb = xlrd.open_workbook(loc_d)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
dictionary_out=np.zeros((sheet.nrows,sheet.ncols)); 
for k in range(sheet.nrows):
    for m in range(sheet.ncols):
        dictionary_out[k][m]=sheet.cell_value(k,m);






##################DWT of Temperature perturbations#####################
#######################################################################


import matplotlib.pyplot as plt
import pywt
import sys
import numpy as np
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from random import *
from pandas import ExcelWriter
from pandas import ExcelFile
from math import log 
from scipy.signal import savgol_filter


coeff_0=np.zeros((7,51));
coeff_1=np.zeros((7,51));
coeff_2=np.zeros((8,51));
coeff_3=np.zeros((9,51));
coeff_4=np.zeros((12,51));
coeff_5=np.zeros((18,51));
coeff_6=np.zeros((29,51));
# Data format:
# Raw data should be in a .txt file with two columns, separated by tabs:
#  - The first column should be a time-series index
#  - The second column should contain the data to be filtered

# Get data:


coeff_zero=np.zeros((7,51));
#------------------------------Discrete wavelet Transform  for fixed basis--------------
for j in range(1,51,1):
    w=pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(learning_data[1:168][j]), w.dec_len)
    maxlev = 6 # Override if desired
    print("maximum level is " + str(maxlev))
    threshold = .0001 # Threshold for filtering
    
    # Decompose into wavelet components, to the level selected:
    coeffs =( pywt.wavedec(learning_data[1:168][j], 'sym4', level=maxlev))
    coeffs_svd =( pywt.wavedec(learning_data[1:168][j], 'sym4', level=maxlev))
    #cA = pywt.threshold(cA, threshold*max(cA))
    plt.figure(1)
    plt.figure(figsize=(12,10));
    for i in range(1, len(coeffs)):
        plt.subplot(maxlev, 1, i)
        plt.plot(1*coeffs[i],color='r',label='original')
        coeffs[i] =pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
        plt.plot(coeffs[i],color='b',label='Denoised')
        #plt.ylim(-15,15)
        plt.grid(True)
        plt.hold(True)
    plt.savefig('dwt_cofficients_temp_perturbations_22_Mar_2000_1'+str(j)+'_.pdf',dpi=1200);
    plt.show();
    #--------------------------------------------------------
    #---------------writing cofficients to file-----------------
    #------------------------------------------------------------
        
    for u in range(0,6,1):
        coeff_0[u][j]=coeffs[u][0];
        
        
    for u in range(0,6,1):
        coeff_1[u][j]=coeffs[u][1];
        
        
    for u in range(0,7,1):
        coeff_2[u][j]=coeffs[u][2];
        
    for u in range(1,8,1):
        coeff_3[u][j]=coeffs[u][3];
            
    for u in range(0,11,1):
        coeff_4[u][j]=coeffs[u][4];
            
    for u in range(0,17,1):
        coeff_5[u][j]=coeffs[u][5];
        
    for u in range(0,28,1):
        coeff_6[u][j]=coeffs[u][6];
        
df = pd.DataFrame(coeff_zero)
df.to_excel(excel_writer = "G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3_study_of_gravity_waves_with_rayleigh_lidar/working_programs/programs/ksvd_GW_analysis/dwt_coefficients_temp_perturbations_22_mar_2000__coeff_1_journal.xlsx")










