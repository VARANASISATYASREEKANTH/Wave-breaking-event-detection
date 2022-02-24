clc; 
clear all; 
close all;
AA=xlsread('test_data_21_10_1998_ver2_gradient.xlsx');
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
N_square=(9.8.*grad_temp)./T_0_dash;


figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1]
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [20 20]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 20 20]);

subplot(2,2,1)
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


%------Breaking Probability P_B----------------
P_B=[0,
0.0980257,
0.0449701	,
0.0945853	,
0.0434346	,
0.174806	,
0.111581	,
0.12094	,
0.0109398	,
0.0424464	,
0.0785493	,
0.0811085	,
0.074558	,
0,
0.0897163	,
0.104128	,
0.105099	,
0.122291	,
0.0687708	,
0.0762099	,
0.0394616	,
0.134548	,
0.0831972	,
0.140018	,
0.0367748	,
0.0943436	,
0.0852295	,
0.0920916	,
0.0972462	,
0.0888307	,
0.0828909	,
0.0621753	,
0.0732549	,
0.0428067	,
0.109393	,
0.0323386	,
0.0475991	,
0.0955909	,
0.0850177	,
0.101375	,
0.0947964	,
0.0980538	,
0.0590061	,
0.129101	,
0.122252	,
0.0903602	,
0.0762995	,
0.0707563	,
0.102297	,
0.0508426	,
0.0834231	,
0.0698597	,
0.0722811	,
0.0479657	,
0.0827935	,
0.111328	,
0.0779245	,
0.0585203	,
0.0644048	,
0.0677394	,
0.102043	,
0.115116	,
0.160127	,
0.0722614	,
0.11471	,
0.0944791	,
0.0817145	,
0.0353904	,
0.0763141	,
0	,
0.0991493	,
0.0345387	,
0.0903073	,
0.114178	,
0	,
0.0547959	,
0.118305	,
0.114917	,
0.0983179	,
0.115531	,
0.106741	,
0.0886875	,
0.0903955	,
0.0829748	,
0.103227	,
0.0937857	,
0.0807286	,
0.0764407	,
0.0451365	,
0.0525631	,
0.0868893	,
0.066116	,
0.581103	,
0.0761954	,
0.132548	,
0.115515	,
0.133715	,
0.0680975	,
0.129339	,
0.109198	,
0.0839114	,
0.101744	,
0.114445	,
0.0865159	,
0.0448992	,
0.0337661	,
0.0623496	,
0.0731622	,
0.087887	,
0.112981	,
1.37545	,
0.0495474	,
0.107065	,
0.059072	,
0.134419	,
0.0567936	,
0.11557	,
0.0851248	,
0.0554358	,
0.0357671	,
0.0621652	,
0.0561259	,
0.124657	,
0.111981	,
0.109461	,
0.0907936	,
0.0567686	,
0.0667977	
];
k=transpose(N_square(1,57:184));
scatter((P_B(1:128,1)./max(P_B(1:128,1))),(k));






