% clear all;
% parallel_reservoir_benchmarking_GradientDescent();

clear all;
% KS L44 jobid=1
Copy_6_of_parallel_reservoir_benchmarking_forKS();

clear all;
% KS L44 jobid=1
Copy_8_of_parallel_reservoir_benchmarking_forKS();

% clear all;
% % KS L22 訓練時間20001でlocality=60:100, jobid=121:130をやる
% Copy_2_of_parallel_reservoir_benchmarking_forKS();
% 
% clear all;
% % KS L22 g=6, train_steps=2e4, 8e4 uniform_fix (130) locality=60:100, jobid=1:10
% Copy_7_of_parallel_reservoir_benchmarking_forKS();

% clear all;
% % parallel fix number of all nodes KS_slow L22
% Copy_3_of_parallel_reservoir_benchmarking_forKS();
% 
% clear all;
% % parallel fix number of all nodes KS_slow L22
% Copy_5_of_parallel_reservoir_benchmarking_forKS();

% clear all;
% % parallel fix number of all nodes vary locality by 1 
% Copy_of_parallel_reservoir_benchmarking_forKS();


% clear all;
% % create KS_slow L=44 data
% kursiv_data();
% 
% close all;
% 
% clear all;
% % parallel fix number of nodes in one reservoir KS_slow L44
% Copy_of_parallel_reservoir_benchmarking_forKS();
% 
% clear all;
% % parallel fix number of all nodes KS_slow L44
% Copy_3_of_parallel_reservoir_benchmarking_forKS();


% 
% clear all;
% % parallel fix number of nodes in one reservoir KS_slow L22 
% Copy_of_parallel_reservoir_benchmarking_forKS();
% 
% clear all;
% % parallel fix number of nodes in one reservoir KS L22
% parallel_reservoir_benchmarking_forKS();


% clear all;
% % parallel fix number of nodes in one reservoir KS_slow L22 
% Copy_of_parallel_reservoir_benchmarking_forKS();
% 
% clear all;
% % parallel fix number of nodes in one reservoir KS L22
% parallel_reservoir_benchmarking_forKS();

% % spatial L22
% spatial_reservoir_benchmarking_forKS();
% 
% clear all;
% % parallel and spatial L22
% parallel_reservoir_benchmarking_forKS();

% clear all;
% % spatial L 44
% Copy_spatial_reservoir_benchmarking_forKS();

% clear all;
% % parallel fix number of nodes in one reservoir L22 
% Copy_of_parallel_reservoir_benchmarking_forKS();