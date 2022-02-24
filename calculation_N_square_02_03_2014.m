clc; 
clear all; 
close all;
AA=xlsread('test_data_21_02_2007.xlsx');
%AA=AA1(1:184,1:61);
%AA_CNN=xlsread('cnn_pert.xlsx');
ht=AA(:,1);%%%height
AA(:,1)=[];%%%height in first column is removing
time=0:4:236;%%%time in minute
ht1=25.1:0.3:80;%%%% height
%%%%plotting temperature data




%plot of potential temperature
r_e=6378.14;% radius of earth
T_0_dash=mean(transpose(AA(1:184,1:30)));
T_0=movmean(T_0_dash,8);
grad_temp =(gradient(T_0,0.3));
g_z=9.8.*(1-(2*ht)./r_e);
c_p=100.0035;


N_square=(4.*3.14.*3.14.*(g_z./T_0_dash).*(grad_temp+(100*g_z./c_p)));

figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [20 10]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 20 10]);

subplot(1,2,1)
plot(-grad_temp,ht1,'b','linewidth',3);
hold on
plot(10*ones(184),ht1,'r--','linewidth',3);
ylim([50 80]);
xticks([-20:5:20])
xlim([-20 20]);
set(gca,'linewidth',2,'fontsize',24);
%title('Gradient of Potential temperature');
set(gca,'Fontweight');
ylabel('Altitude(km)');
xlabel('d\theta/dz(K/km)');


subplot(1,2,2)
plot((N_square(184,1:184)),ht1,'b','linewidth',3);
hold on
plot(zeros(184),ht1,'r--','linewidth',3);
ylim([50 80]);
xticks([-20:5:20])
xlim([-20 20]);
set(gca,'linewidth',2,'fontsize',24);
%title('');
set(gca,'Fontweight');
ylabel('Altitude(km)');
xlabel('N^2x10^{-4}');






