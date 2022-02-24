clc; 
clear all; 
close all;
AA=xlsread('test_data_21_04_2014_ver3.xlsx');
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
set(gcf, 'PaperSize', [30 20]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 30 20]);

subplot(221)
contourf(time+4,ht1,AA(1:184,:),'linestyle','none','levelstep',1);
colormap('jet');
caxis([140 300]);
colorbar;
xticks([0:60:240])
xlim([0 240]);
ylim([30 80]);
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
[p2 s]=polyfit(ht1,TA2,2);
polu2=polyval(p2,ht1);
clear p2; clear s;
Ttt(:,i)=polu2;
Tp2(:,i)=TA2-polu2';
clear polu2 TA2;
end
%%%plotting obtained perturbation
subplot(222)
contourf(time+4,ht1,Tp2,'linestyle','none','levelstep',0.1);
colormap('jet');
set(gca,'linewidth',2,'fontsize',28);
xticks([0:60:240])
xlim([0 240]);
caxis([-10 10]);
colorbar;
ylim([30 80]);
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
ylim([30 80]);
xticks([0:60:240])
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
xticks([0:60:240])
xlim([0 240]);
set(gca,'linewidth',2,'fontsize',28);
ylabel('Vertical wavelength(km)');
xlabel('Time(min)');
title('Amplitude of Wavenumber spectrum (K)');
set(gca,'Fontweight','bold');
hcb.Title.String = "";
%%%%%%bpf 5-8 km bandpass filtering 
     pr1=2.5;pr2=8;%--------------------------------------------------------------------------------
    fcut1=1./pr2;fcut2=1./pr1;
    [a,b]=butter(2,[(2*fcut1)/smf  (2*fcut2)/smf],'Bandpass');
for i=1:length(time)
TQB(:,i)=filter(a,b,Tp2(:,i));

end

%[cA,cD] = dwt(Tp2,'sym4');


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
caxis([-2 2]);
colorbar;
ylim([60 80]);
xticks([0:60:240])
xlim([0 60]);
set(gca,'linewidth',2,'fontsize',24);
title('\lambda_z=[4.5km, 4.8km]');
set(gca,'Fontweight','bold');
ylabel('Altitude(km)');
xlabel('Time(min)');
hcb.Title.String = "";
%%%%%15-40min bandpass filtering
    pr1=15;pr2=45%--------------------------------------------------------------------------------
    fcut1=1./pr2;fcut2=1./pr1;
    [a,b]=butter(2,[(2*fcut1)/smf  (2*fcut2)/smf],'Bandpass');
for i=1:length(ht1)
TQA(i,:)=filter(a,b,TQB(i,:));

end
%%%%%%%5-8km band pass filterd+15-40min bandpass filtering 
subplot(212)
contourf(time+4,ht1,TQA,'linestyle','none','levelstep',0.1);
colormap('jet');
caxis([-1 1]);
colorbar;
ylim([30 80]);
ylabel('Altitude(km)');
xlabel('Time(min)');
xticks([0:60:240])
xlim([0 240]);
set(gca,'linewidth',2,'fontsize',24);
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









%plot of potential temperature
T_0_dash=mean(transpose(AA));
T_0=movmean(T_0_dash,1);
grad_temp = (gradient(T_0,0.3)+9.8/(1.001));
N_square=(9.8./T_0).*grad_temp;



figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [20 20]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 20 20]);

subplot(2,1,1)
plot(grad_temp,ht1,'b','linewidth',3);
hold on
plot(zeros(184),ht1,'r--','linewidth',3);
ylim([60 80]);
%xticks([50:60:240])
xlim([-10 40]);
set(gca,'linewidth',2,'fontsize',24);
title('Gradient of Potential temperature');
set(gca,'Fontweight');
ylabel('Altitude(km)');
xlabel('d\theta/dz(K/km)');


%vertical wavenumber vs height
k=abs(fft(transpose(movmean(Tp1,5))));









