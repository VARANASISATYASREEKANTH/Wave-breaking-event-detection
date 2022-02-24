# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:23:28 2019

@author: asdg
"""

#coding:latin_1
#coding:latin_1

from Functions import *
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 
import xlrd
#-------------------------------------------------------------------------------------------------------------------#
#-------------------------------------------------- 1. Initial settings. -------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

resize_shape = (2,2)  # Resized image's shape
sigma =5            # Noise standard dev.

window_shape = (32, 32)    # Patches' shape
step = 5             # Patches' step
ratio =0.82            # Ratio for the dictionary (training set).
ksvd_iter =10            # Number of iterations for the K-SVD.

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------- 3. Image processing. --------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
#test_data=[[18335,17886,17812,16339],[16998,16527,16589,15168],[15656,15223,15026,13896],[14619,14187,14248,13223]]
#training_data=[[136386,136791,129025,123170],[129912,136793,122619,116456],[121105,122434,114544,109969],[113541,115009,10742,102347]]

#--------------reading training data-------------------------

loc_d = ('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN//tr_data_pert_april_2014_mls_64_samples.xlsx');#train data
wb = xlrd.open_workbook(loc_d)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
training_data= np.zeros(shape=(sheet.nrows,sheet.ncols))
for k in range(1,sheet.ncols):#columns
    for i in range(0,sheet.nrows):#rows
       training_data[i][k]=sheet.cell_value(i,k);
#--------------reading testing  data-------------------------
test_data= np.zeros(shape=(sheet.nrows, sheet.ncols))
loc_d = ('G:/research_works_vssreekanth_jrf/MY_PAPERS/paper_3a_deep_learning_for_parameter_estimation/programs/anamoly_detection_LSTM_RNN/tst_data_21_04_2014_64_samples.xlsx');#test data
wb = xlrd.open_workbook(loc_d)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
for k in range(1,sheet.ncols):#columns
    for i in range(1,sheet.nrows):#rows
       test_data[i][k]=sheet.cell_value(i,k);



noisy_data =test_data;


#-------------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------- 4. Denoising. -----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

denoised_data, calc_time, n_total = denoising(noisy_data,test_data, window_shape, step, sigma, ratio, ksvd_iter)



#--------------------------inverse discrete wavelet transform----------------------------
''''
wb1 = Workbook() 
    # add_sheet is used to create sheet. 
sheet2= wb1.add_sheet('Sheet1')



for w in range(1,2,1):
    
    #----------------------r-channel---------------------------------#
    loc1 = ('G:/lidar_data/DATA2014/20140130/r700.xlsx')#test_data
     
    wb = xlrd.open_workbook(loc1)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)
     
    for i in range(sheet.nrows):
       r_channel_photon_count[i]=sheet.cell_value(i,w);
       counter=i;
    data=r_channel_photon_count
    cA,cD=pywt.dwt(data,'sym4')
    #------------------------------i
plt.figure(3)

fig = plt.figure(figsize=(7,5));

ax = plt.subplot(111)
    

ax.plot(vertical_height[1:513],(cD), label='wavelet detail cofficients',linewidth='1.7',color='C0');
A= pywt.idwt(cA, cD_T, 'db1', 'smooth')  

plt.yscale('log')
plt.xlim(0,120);
plt.ylim(1,10e6);
plt.ylabel('Signal to Noise Ratio');
plt.xlabel('Height(KM)');
plt.title('30 Jan 2014');
plt.grid(True);
ax.legend()
plt.savefig('dwt_30_jan_2014.eps',dpi=1200);
plt.savefig('dwt_30_jan_2014.pdf',dpi=1200);

plt.show()



plt.figure(4)

fig = plt.figure(figsize=(7,5));

ax = plt.subplot(111)
    

ax.plot(vertical_height,A, label='reconstructed',linewidth='1.7',color='k');
ax.plot(vertical_height,u_channel_photon_count, label='original',linewidth='1.7',color='r');

plt.yscale('log')
plt.xlim(0,120);
#plt.ylim(1,10e2);
plt.ylabel('reconstructed');
plt.xlabel('Height(KM)');
plt.title('30 Jan 2014');
plt.xlim(20,120);
plt.grid(True);
ax.legend()
plt.savefig('idwt_30_jan_2014.eps',dpi=1200);
plt.savefig('idwt_30_jan_2014.pdf',dpi=1200);

plt.show()
#dwt--for r -channel-------------------- 
   
    #-----------------------------------------------------------------------
    datarec = pywt.waverec(coeffs, 'sym4')
    
    
    t_min = 90
    t_max= 400
    plt.figure(3)
   
    plt.figure(figsize=(8,3));
    plt.plot(vertical_height[t_min:t_max], data[t_min:t_max],label='Original',linewidth='1.8',color='r')
    plt.plot(vertical_height[t_min:t_max], datarec[t_min:t_max],label='reconstructed',linewidth='1.8',color='b')
    plt.xlabel('time (milli seconds)')
    plt.ylabel('Photon Counts')
    plt.xlim(20,120)
    plt.ylim(10E-1,10e6)
    plt.legend()
    plt.yscale('log')
    plt.title("De-noised signal using Discrete Wavelet \n  Transform(Symlet4) technique")
    plt.grid(True)
   
    plt.savefig('Wavelet_denoising_using_Symlet4.pdf',dpi=1200);
    plt.show()

    for u in range(0,1024,1):
        if(cD[u]<0):
            cD[u]=0;
        else:
        # u=columns,w=rows
        sheet1.write(u,a , combined_temperature[u]) 
        sheet1.write(u,b , vertical_height[u]) 
        sheet1.write(u,c , standard_error[u]) 
        wb.save('DI_Temperature_11_jan_1999_SVD_synthetic'+str(w)+'.xls')
        sheet2.write(u,w,datarec)

        wb1.save('G:/research_works_vssreekanth_jrf/MY_PAPERS/sparsity_including_norms_on_frame_expansion/programs/compressive_sensing_sparsity/K-SVD-master___working_code_for_updation/detail_cofficients_test_30_jan_2014.xlsx') 
'''

