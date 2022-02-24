clear all;
close all;
[d_tst1,txt,raw]=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\Aura_MLS\AURA_DATA_April_2014\mls_aura_test_data_21_04_2014.xlsx');
[d_tst2,txt,raw]=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\Aura_MLS\AURA_DATA_April_2014\mls_aura_test_data_07_04_2014.xlsx');
[d_tst3,txt,raw]=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\Aura_MLS\09_01_2015\test_data_09_01_2015_mls_1.xlsx');
[d_tr,txt,raw]=xlsread('G:\research_works_vssreekanth_jrf\MY_PAPERS\paper_3a_deep_learning_for_parameter_estimation\Aura_MLS\AURA_DATA_April_2014\mls_aura_training_data_april_2014.xlsx');
%----train data
d_tr_pert=transpose(d_tr-mean(d_tr));
d_tr_per_re=resample(d_tr_pert,8,2);
train=resample(d_tr_per_re,20,4);
final_train=transpose(train);

%---tst1 data
d_tst1_pert=transpose(d_tst1-mean(d_tst1));
tst1_re=resample(d_tst1_pert,10,4);
tst1_re2=resample(transpose(tst1_re),23,2);
final_tst1=transpose(tst1_re2);

%---tst2 data
d_tst2_pert=transpose(d_tst2-mean(d_tst2));
tst2_re=resample(d_tst2_pert,10,4);
tst2_re2=resample(transpose(tst2_re),23,2);
final_tst2=transpose(tst2_re2);

%---tst3 data
d_tst3_pert=transpose(d_tst3-mean(d_tst3));
tst3_re=resample(d_tst3_pert,10,4);
tst3_re2=resample(transpose(tst3_re),30,2);
final_tst3=transpose(tst3_re2);