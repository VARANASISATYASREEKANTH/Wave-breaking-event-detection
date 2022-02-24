clear all;
close all;
[data,txt,raw]=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\Aura_MLS\21_04_2014\training_data_21_04_2014.xlsx');
AA=transpose(data);
AA(:,1)=[];%%%height in first column is removing
time=1:1:48929;%%%time in minute
ht=10:2:118;%%%% height
ht_T=transpose(ht);
%%%%plotting temperature data


figure
fig=gcf;
fig.Units='normalized';
fig.OuterPosition=[0 0 1 1];
set(gcf, 'PaperUnits', 'inches');
set(gcf, 'PaperSize', [30 20]);
set(gcf, 'PaperPositionMode', 'auto');
set(gcf, 'PaperPosition', [0 0 30 20]);
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
subplot(2,4,1)
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
for i=1:length(time)
TA2=Tp1(1:55,i);
[p2 , ~]=polyfit(ht,TA2,2);
polu2=polyval(p2,ht);
clear p2; clear s;
Ttt(:,i)=polu2;
Tp2(:,i)=TA2-polu2';
clear polu2 TA2;
end

Tp2_resampled=resample(Tp2(:,10001:16000), 20,3);
%%%plotting obtained perturbation
%subplot(222)
%contourf(time,ht,Tp2,'linestyle','none','levelstep',0.1);
%colormap('hsv');
%set(gca,'linewidth',2,'fontsize',20);
%xticks([0:60:240])
%xlim([0 240]);
%caxis([-10 10]);
%colorbar;
%ylim([30 80]);
%title('Temperature perturbation (K)');
%ylabel('Altitude (km)');
%xlabel('Time(min)');
%set(gca,'Fontweight','bold');
%############################################################################################
%#######################################resampling###########################################
%xlswrite('mls_aura_train_data_perturbations.xlsx', Tp2);
