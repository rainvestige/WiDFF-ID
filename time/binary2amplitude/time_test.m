clear; clc;

abs_dat_name = 'NO20.dat';
num_iteration = 100
total_time = 0;

for i = 1:num_iteration
    id = tic; % start timer.

    csi_trace = read_bf_file(abs_dat_name);
    num_packets = length(csi_trace);
    num_rx = csi_trace{1}.Nrx;
    num_tx = csi_trace{1}.Ntx;
    num_subcarriers = size(csi_trace{1}.csi)(3);

    csi_array = zeros([num_packets, num_tx, num_rx, num_subcarriers]);

    for i = 1 : num_packets
        csi = get_scaled_csi(csi_trace{i});  % (num_tx * num_rx * num_subcarriers)
        csi = abs(csi); % extract the amplitude.
        csi_array(i, :, :, :) = csi; % (N * num_tx * num_rx * num_subcarriers)
    end

    elapsed_time = toc(id); % show the elapsed time.
    total_time += elapsed_time;
end


result = (total_time / num_iteration) / num_packets * 50;
result = result * 1e3; % Convert to ms.
printf(...
    'Time taken to convert 50 CSI samples from binary data to magnitude data: %f ms\n', result); 
