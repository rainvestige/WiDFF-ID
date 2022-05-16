close all; clear;

pkg load signal;

num_iteration = 100;
total_time = 0;
selected_samples = 50; % Only need 50 samples in test stage to determine the identity.

%% Define the butterworth LPF
for i = 1:num_iteration
    id = tic;
    %% Load the data.
    fname = ...
    '/home/public/b509/code/g19/xxy/projects/CSI-DenseNet/data/20201120/NO17.mat';
    load(fname);
    [b, a] = butter(5, 2/100);
    flt_data = filter(b, a, csi_array);
    elapsed_time = toc(id);
    total_time += elapsed_time;
end

num_samples = length(csi_array);
result = (total_time / num_iteration) / num_samples * 50;

printf('Time spent filtering 50 CSI samples: %fms\n', result*1000);
filtered_data = flt_data(301:350,:,:,:);
save('-v6', 'filtered_data.mat', 'filtered_data');

