clc;
clear all;
close all;
%--------------------------------------------------
[d_64,txt,raw]=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\Aura_MLS\AURA_DATA_April_2014\D_21_04_2014_1024_samples.xlsx');
[x_64,txt,raw]=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\Aura_MLS\AURA_DATA_April_2014\X_21_04_2014_1024_samples.xlsx');
d_64_1=resample(d_64,1,10); d_64_2=resample(transpose(d_64_1),1,10); d_64_final=transpose(d_64_2);
x_64_1=resample(x_64,1,10); x_64_2=resample(transpose(x_64_1),1,10); x_64_final=transpose(x_64_2);
y=d_64_final*x_64_final;
y1=resample((y),10,8);
y2=resample(transpose(y1),22,1);
y3=transpose(y2);





