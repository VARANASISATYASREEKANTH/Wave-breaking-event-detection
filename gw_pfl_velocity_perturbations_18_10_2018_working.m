clc; 
clear all; 
close all;
AA=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\programs\anamoly_detection_LSTM_RNN\meridonal_wind_18_10_2018.xlsx');

ht=AA(:,1);%%%height
AA(:,1)=[];%%%height in first column is removing
time=0:4.2:588;%%%time in minute
ht1=80.013:0.03:122.613;%%%% height
%%%%plotting temperature data
figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [30 20]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 30 20]);

subplot(221)
contourf(time(1:141),ht1(1:1421),AA(1:1421,1:141),'linestyle','none','levelstep',1);
colormap('jet');
caxis([-50 50]);
colorbar;

xlim([0 600]);
ylim([80 100]);
title('Temperature (K)');
ylabel('Altitude (km)');
xlabel('Time(min)');
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',28);
hcb.Title.String = "";
%%%%considering 30-60km for gw analysis
%%%%second order polynomial fit in timewise for each height
for i=1:length(ht)
    TA2=AA(i,:);
[p2 s]=polyfit(time(1:60),TA2,2);
polu2=polyval(p2,time(1:60));
clear p2; clear s;
Tp1(i,:)=TA2-polu2;
clear polu2 TA2;   
end
%%%%second order polynomial fit in heightwise for each time
for i=1:length(time(1:60))
TA2=Tp1(1:499,i);
[p2 s]=polyfit(ht1,TA2,2);
polu2=polyval(p2,ht1);
clear p2; clear s;
Ttt(:,i)=polu2;
Tp2(:,i)=TA2-polu2';
clear polu2 TA2;
end
%%%plotting obtained perturbation
subplot(222)
contourf(time(1:60),ht1(1:499),Tp2(1:499,1:60),'linestyle','none','levelstep',0.1);
colormap('jet');
set(gca,'linewidth',2,'fontsize',28);
%xticks([0:90:360])
%xlim([0 360]);
caxis([-40 40]);
colorbar;
%ylim([30 80]);
title('Temperature perturbation (K)');
ylabel('Altitude (km)');
xlabel('Time(min)');
set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',28);
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
subplot(223)
contourf(1./f(2:end),ht1,2.*abs(YY1(:,2:end)),'linestyle','none','levelstep',0.1);
colormap('jet');
caxis([0 20]);
colorbar;
ylim([80 90]);
xticks([0:60:252])
xlim([0 240]);

%xlim([8 20]);
title('Amplitude of Frequency Spectrum (K)');
ylabel('Altitude (km)');
xlabel('Period(min)');

set(gca,'Fontweight','bold');
set(gca,'linewidth',2,'fontsize',28);
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
subplot(224)
contourf(time+4,1./f(2:end),2.*abs(YY2(:,2:end))','linestyle','none','levelstep',0.1);
colormap('jet');
caxis([0 4]);
colorbar;
ylim([0 15]);
xticks([0:90:360])
xlim([0 360]);
set(gca,'linewidth',2,'fontsize',28);
ylabel('Vertical wavelength(km)');
xlabel('Time(min)');
title('Amplitude of Wavenumber spectrum (K)');
set(gca,'Fontweight','bold');
hcb.Title.String = "";
%%%%%%bpf 5-8 km bandpass filtering 
     pr1=10;pr2= 10.5;%--------------------------------------------------------------------------------
    fcut1=1./pr2;fcut2=1./pr1;
    [a,b]=butter(2,[(2*fcut1)/smf  (2*fcut2)/smf],'Bandpass');
for i=1:length(time)
TQB(:,i)=filter(a,b,Tp2(:,i));

end




%%%%5-8km band pass filterd
figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [20 30]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 20 30]);

subplot(211)
contourf(time+4,ht1,TQB,'linestyle','none','levelstep',0.1);
colormap('jet');
caxis([-4 4]);
colorbar;
ylim([80 95]);
xticks([0:60:240])
xlim([0 240]);
set(gca,'linewidth',2,'fontsize',24);
title('\lambda_z=[4.5km, 4.8km]');
set(gca,'Fontweight','bold');
ylabel('Altitude(km)');
xlabel('Time(min)');
hcb.Title.String = "";
%%%%%15-40min bandpass filtering
    pr1=15;pr2=30;%--------------------------------------------------------------------------------
    fcut1=1./pr2;fcut2=1./pr1;
    [a,b]=butter(2,[(2*fcut1)/smf  (2*fcut2)/smf],'Bandpass');
for i=1:length(ht1)
TQA(i,:)=filter(a,b,TQB(i,:));

end
%%%%%%%5-8km band pass filterd+15-40min bandpass filtering 
subplot(212)
contourf(time+4,ht1,TQA,'linestyle','none','levelstep',0.1);
colormap('jet');
caxis([-0.4 0.4]);
colorbar;
ylim([80 95]);
ylabel('Altitude(km)');
xlabel('Time(min)');
xticks([0:60:240])
xlim([0 240]);
set(gca,'linewidth',2,'fontsize',24);
title('\lambda_z=[4.5km, 4.8km],T=[15min, 80min]');
set(gca,'Fontweight','bold');
hcb.Title.String = "";

%---------------------------------------------------------------------------



