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
import pandas as pd
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 1. Initial settings. -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
resize_shape = (8,8)  # Resized image's shape
sigma=5               # Noise standard dev.
window_shape = (1, 128) # Patches' shape
step = 21                # Patches' step
ratio =3.35# Ratio for the dictionary (training set).
ksvd_iter =50              # Number of iterations for the K-SVD.
#-------------------------------------------------------------------------------------------------------------------#
#--------------------------------------------------- 2. data import ----------------------------------------------#
#-----------------------------------------------photon count data--------------------------------------------------------------------#

loc_d = ('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/test_data/21_oct_1998/y_out_cnn_21_10_1998_dl_train_data.xlsx')

wb = xlrd.open_workbook(loc_d)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
learning_data=np.zeros((sheet.nrows,sheet.ncols));  
for k in range(sheet.nrows):
    for m in range(sheet.ncols):
        learning_data[k][m]=sheet.cell_value(k,m);


loc_d = ('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/test_data/21_oct_1998/y_out_cnn_21_10_1998_dl_test_data.xlsx')
wb = xlrd.open_workbook(loc_d)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
testing_data=np.zeros((sheet.nrows,sheet.ncols)); 
for k in range(sheet.nrows):
    for m in range(sheet.ncols):
        testing_data[k][m]=sheet.cell_value(k,m);

#-------signal denoising------
denoised_data, calc_time, n_total = denoising(testing_data, learning_data, window_shape, step, sigma, ratio, ksvd_iter);

