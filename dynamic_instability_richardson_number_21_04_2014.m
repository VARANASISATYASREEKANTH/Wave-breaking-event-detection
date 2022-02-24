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
figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1];
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [35 18]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 35 18]);
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
subplot(2,4,1)
contourf(time+4,ht1,AA(1:184,:),'linestyle','none','levelstep',1);
colormap('hsv');
caxis([140 300]);
colorbar;
xticks([0:60:240]);
xlim([0 240]);
ylim([30 80]);
title('Temperature (K)');
ylabel('Altitude (km)');
xlabel('Time(min)');
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',20);
hcb.Title.String = "";
%%%%considering 30-60km for gw analysis
%%%%second order polynomial fit in timewise for each height
for i=1:length(ht)
    TA2=AA(i,:);
[p2 s]=polyfit(time+4,TA2,2);
polu2=polyval(p2,time);
clear p2; clear s;
Tp1(i,:)=TA2-polu2;
clear polu2 TA2;   
end
%%%%second order polynomial fit in heightwise for each time

ht1=25.1:0.3:80;
for i=1:length(time)
TA2=Tp1(1:184,i);
[p2 , ~]=polyfit(ht1,TA2,2);
polu2=polyval(p2,ht1);
clear p2; clear s;
Ttt(:,i)=polu2;
Tp2(:,i)=TA2-polu2';
clear polu2 TA2;
end
%%%plotting obtained perturbation
%subplot(222)
%contourf(time+4,ht1,Tp2,'linestyle','none','levelstep',0.1);
colormap('hsv');
set(gca,'linewidth',2,'fontsize',20);
xticks([0:60:240])
xlim([0 240]);
%caxis([-10 10]);
colorbar;
ylim([30 80]);
title('Temperature perturbation (K)');
ylabel('Altitude (km)');
xlabel('Time(min)');
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',20);
hcb.Title.String = "";
%%% doing fft analysis in timewise
smt=4;%%%minute
smf=1/smt;
for i=1:length(ht1)%%%%heightwise
xx=length(Tp2(i,:));
TT = (0:xx-1)*smt;    
NFFT = 2^nextpow2(xx); % Next power of 2 from length of y
Y1 = fft(Tp2(i,:),NFFT)/xx;
f= smf/2*linspace(0,1,NFFT/2+1);
pr=1./f;
Y1=Y1';
f= smf/2*linspace(0,1,NFFT/2+1);
pr=1./f;
YY1(i,:)=Y1(1:length(f));
end
%%%%% plotting frequency spectra
subplot(2,4,2)
contourf(1./f(2:end),ht1,2.*abs(YY1(:,2:end)),'linestyle','none','levelstep',0.1);
colormap('hsv');
caxis([0 20]);
colorbar;
ylim([30 80]);
xticks([0:60:240])
xlim([0 240]);

%xlim([8 20]);
title('Amplitude of Frequency Spectrum (K)');
ylabel('Altitude (km)');
xlabel('Period(min)');

set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',20);
hcb.Title.String = "";
%%%doing fft in height domain
smt=0.3;%%%in km
smf=1/smt;
for i=1:length(time)%%%%timewise
xx=length(Tp2(:,i));
TT = (0:xx-1)*smt;    
NFFT = 2^nextpow2(xx); % Next power of 2 from length of y
Y1 = fft(Tp2(i,:),NFFT)/xx;
f= smf/2*linspace(0,1,NFFT/2+1);
pr=1./f;
Y1=Y1';
f= smf/2*linspace(0,1,NFFT/2+1);
pr=1./f;
YY2(i,:)=Y1(1:length(f));
end
%%%%%%plotting wavenumber spectra
subplot(2,4,3)
contourf(time+4,1./f(2:end),2.*abs(YY2(:,2:end))','linestyle','none','levelstep',0.1);
colormap('hsv');
caxis([0 4]);
colorbar;
ylim([0 15]);
xticks([0:60:240])
xlim([0 240]);
set(gca,'linewidth',2,'fontsize',20);
ylabel('Vertical wavelength(km)');
xlabel('Time(min)');
title('Amplitude of Wavenumber spectrum (K)');
set(gca,'Fontweight','bold');
hcb.Title.String = "";
%%%%%%bpf 5-8 km bandpass filtering 
     pr1=6;pr2=10;%--------------------------------------------------------------------------------
    fcut1=1./pr2;fcut2=1./pr1;
    [a,b]=butter(2,[(2*fcut1)/smf  (2*fcut2)/smf],'Bandpass');
for i=1:length(time)
TQB(:,i)=filter(a,b,Tp2(:,i));
end

%[cA,cD] = dwt(Tp2,'sym4');


%%%%5-8km band pass filterd
subplot(2,4,4)
contourf(time+4,ht1,TQB,'linestyle','none','levelstep',0.1);
colormap('hsv');
%caxis([-2 2]);
colorbar;
ylim([30 80]);
xticks([0:60:240])
xlim([0 240]);
set(gca,'linewidth',2,'fontsize',20);
title('\lambda_z=[4.5km, 4.8km]');
set(gca,'Fontweight','bold');
ylabel('Altitude(km)');
xlabel('Time(min)');
hcb.Title.String = "";
%%%%%15-40min bandpass filtering
    pr1=10;pr2=50%--------------------------------------------------------------------------------
    fcut1=1./pr2;fcut2=1./pr1;
    [a,b]=butter(2,[(2*fcut1)/smf  (2*fcut2)/smf],'Bandpass');
for i=1:length(ht1)
TQA(i,:)=filter(a,b,TQB(i,:));

end
%%%%%%%5-8km band pass filterd+15-40min bandpass filtering 
subplot(2,4,5)
contourf(time+4,ht1,TQA,'linestyle','none','levelstep',0.1);
colormap('hsv');
caxis([-2 2]);
colorbar;
ylim([30 80]);
ylabel('Altitude(km)');
xlabel('Time(min)');
xticks([0:60:240])
xlim([0 240]);
set(gca,'linewidth',2,'fontsize',20);
title('\lambda_z=[4.5km, 4.8km],T=[15min, 80min]');
set(gca,'Fontweight','bold');
hcb.Title.String = "";
%---------------------------------------------------------------------------
for i=1:1:184
cA = fft(Tp2(i,:));
CA_T(i,:)=cA;

end
for i=1:1:60
cA = fft(transpose(Tp2(:,i)));
CA_H(i,:)=cA;

end
%-------------------------------------------------------------------------
%-----calculation of potential gradient, N square, potential energy

%
r_e=6378.14;% radius of earth
T_0_dash=mean(transpose(AA(1:184,31:45)));
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
N_square=(4.*3.14.*3.14.*(g_z./T_0_dash).*(grad_temp+(10+g_z./c_p)));
%----calculation of potential energy---
k=g_z.*g_z;
l=mean(transpose((mean(T_pert)./transpose(T_0_dash)).*(mean(T_pert)./transpose(T_0_dash))));
E_p=abs(0.5.*(k./mean(N_square)).*transpose(l));
E_p_mean=(mean(transpose(E_p)))*1e4;



subplot(2,4,7)
plot(-grad_temp,ht1,'k','linewidth',4);
hold on
plot(10*ones(184),ht1,'k--','linewidth',4);
ylim([50 80]);
xticks([-30:10:20])
xlim([-30 20]);
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',20);
ylabel('Altitude(km)');
xlabel('dT/dz(K/km)');


%subplot(2,4,8)
%plot(mean(N_square)/2,ht1,'k','linewidth',3);
%hold on
%plot(zeros(184),ht1,'k--','linewidth',3);
%ylim([50 80]);
%xticks([-10:10:40]);
%xlim([-10 40]);
%set(gca,'Fontweight','bold');
%set(gca,'linewidth',2,'fontsize',18);
%ylabel('Altitude(km)');
%xlabel('N^2x10^{-4} (rad/sec)^2 \newline (b)');

subplot(2,4,8)
plot((log10(E_p_mean)),ht1,'k','linewidth',4);
ylim([50 80]);
%xticks([2:0.1:2.4])
%xlim([2 2.4]);
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',20);
ylabel('Altitude(km)');
xlabel('log_{10}(E_p)(J/Kg)')
%--contourplot of meridonal wind---
V_21_04_2014=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\programs\anamoly_detection_LSTM_RNN\mer_21_04_2014_v2.xlsx');
time_H=0:1:24;
H=70:2:110;
subplot(2,4,6)
contourf(time_H,H,V_21_04_2014,'linestyle','none','levelstep',1);
colormap('hsv');
caxis([-90 90]);
colorbar;
xticks([0:4:24]);
xlim([0 24]);
ylim([70 110]);
title('Meridonal wind(m/s)');
ylabel('Altitude (km)');
xlabel('Time(h)');
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',20);
%--------checking convective and dynamic instability-----







figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1];
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [8 6]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 8 6]);
ax = gca;


%---convective instability----
subplot(1,2,1)
plot(mean(N_square)/2,ht1,'k','linewidth',3);
hold on
plot(zeros(184),ht1,'k--','linewidth',3);
ylim([70 80]);
xticks([-10:10:30]);
xlim([-10 30]);
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',24);
ylabel('Altitude(km)');
xlabel('N^2x10^{-4} (rad/sec)^2');


%---dynamic instability---

%---meridonal velocity---
y1_v = interp(V_21_04_2014(1,1:9),6);
y2_v=interp(V_21_04_2014(1,2:10),6);
y_V=(y2_v-y1_v)/0.3;
y_u=[0.725,-5.17,-14.5,-17.8,-24,-23.7,-14.4,-4,-3.25];
y_u=interp(y_u,6);
y_u1=y_u(1,1:34);y_u2=y_u(1,2:35);y_U=(y_u2-y_u1)/0.3;
%gradient_u=gradient(y,0.6);
%d1=size(N_square(1,95:128)); d2=size(square(gradient_u(1:34)));
R_i=0.65.*(mean(N_square(95:128,:)))./(transpose(square(y_V(1:34)))+transpose(square(y_U(1:34))));

h_R=70.1:0.3:80;
subplot(1,2,2)
plot(transpose(R_i(5,:)),ht1,'k','linewidth',3);
hold on
plot(2.5*ones(34),h_R,'k--','linewidth',3);
ylim([70 80]);
xticks([-10:10:20]);
xlim([-10 20]);
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',24);
ylabel('Altitude(km)');
xlabel('R_i');





