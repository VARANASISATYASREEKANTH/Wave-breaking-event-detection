clc; 
clear all; 
close all;
AA=xlsread('test_data_21_04_2014.xlsx');
%AA=AA1(1:184,1:61);
%AA_CNN=xlsread('cnn_pert.xlsx');
ht=AA(:,1);%%%height
AA(:,1)=[];%%%height in first column is removing
time=0:4:236;%%%time in minute
ht1=25.1:0.3:80;%%%% height
%%%%plotting temperature data




%plot of potential temperature
r_e=6378.14;% radius of earth
T_0_dash=mean(transpose(AA(1:184,15:30)));
T_0=movmean(T_0_dash,8);
grad_temp =(gradient(T_0,0.3));
g_z=9.8.*(1-(2*ht)./r_e);
c_p=100.0035;
T_pert=zeros(184,60);

for i=1:1:184
    for j=1:1:60
        
        T_pert(i,j)=AA(i,j)-T_0_dash(j);
    end
end
N_square=(4.*3.14.*3.14.*(g_z./T_0_dash).*(grad_temp+(100*g_z./c_p)));
%----calculation of potential energy---
k=g_z.*g_z;
l=mean(transpose((mean(T_pert)./transpose(T_0_dash)).*(mean(T_pert)./transpose(T_0_dash))));
E_p=abs(0.5.*(k./mean(N_square)).*transpose(l));
E_p_mean=(mean(transpose(E_p)))*1e4;






figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [21 10]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 21 10]);

subplot(1,3,1)
plot(-grad_temp,ht1,'k','linewidth',3);
hold on
plot(10*ones(184),ht1,'k--','linewidth',3);
ylim([50 80]);
xticks([-30:10:20])
xlim([-30 20]);
set(gca,'linewidth',2,'fontsize',24);
%title('Gradient of Potential temperature');
set(gca,'Fontweight');
ylabel('Altitude(km)');
xlabel('dT/dz(K/km) \newline (a)');


subplot(1,3,2)
plot(mean(N_square)/2,ht1,'k','linewidth',3);
hold on
plot(zeros(184),ht1,'k--','linewidth',3);
ylim([50 80]);
xticks([-10:10:40]);
xlim([-10 40]);
set(gca,'linewidth',2,'fontsize',24);
%title('');
set(gca,'Fontweight');
ylabel('Altitude(km)');
xlabel('N^2x10^{-4} (rad/sec)^2 \newline (b)');


subplot(1,3,3)
plot((log10(E_p_mean)),ht1,'k','linewidth',3);
ylim([50 80]);
%xticks([2:0.1:2.4])
%xlim([2 2.4]);
set(gca,'linewidth',2,'fontsize',24);
set(gca,'Fontweight');
ylabel('Altitude(km)');
xlabel('log_{10}(E_p)(J/Kg)  \newline   (c)');











%

