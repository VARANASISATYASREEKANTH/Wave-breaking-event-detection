clc; 
clear all; 
close all;
AA=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\programs\anamoly_detection_LSTM_RNN\density_estimation_21_04_2014.xlsx');
x_mean=mean(AA);
x_var=var(AA);
x_std=sqrt(x_var);
s=128;% no of samples
bw_opt=1.06*x_std*power(s,-0.2);


figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [10 10]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 10 10]);

[f,xi] = ksdensity(AA(1:128,2),'Bandwidth',1.8);
plot(xi,f,'--r','LineWidth',1.5);




