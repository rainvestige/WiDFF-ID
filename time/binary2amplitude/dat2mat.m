function dat2mat(dat_file, amp_or_angle)
%DAT2MAT Convert the dat format file to the mat format file
%
%   Read in the CSI trace file generated by `log_to_file` command, use
%   `read_bf_file` function to unpack the binary CSI format. The result of
%   `read_bf_file` is a cell array, each cell is one csi packet, it has many
%   fields, the `get_scaled_csi` will use these fields to compute the CSI in
%   absolute units, rather than Intel's internal reference level.

csi_trace = read_bf_file(dat_file);
num_packets = length(csi_trace);
num_rx = csi_trace{1}.Nrx;
num_tx = csi_trace{1}.Ntx;
num_subcarriers = size(csi_trace{1}.csi)(3);

csi_array = zeros([num_packets, num_tx, num_rx, num_subcarriers]);

for i = 1 : num_packets
    csi = get_scaled_csi(csi_trace{i});  % (num_tx * num_rx * num_subcarriers)
    if (amp_or_angle == 0)
        csi = abs(csi); % extract the amplitude.
    elseif (amp_or_angle == 1)
        csi = angle(csi); % extract the angle.
    elseif (amp_or_angle == 2)
        disp(''); % do nothing.
    else
        disp("Invalid value for amp_or_angle, only accep 0 or 1.");
    endif
    csi_array(i, :, :, :) = csi; % (N * num_tx * num_rx * num_subcarriers)
end

if (amp_or_angle == 0)
    mat_file = strcat(dat_file(1 : length(dat_file)-4), '.mat');
elseif (amp_or_angle == 1)
    mat_file = strcat(dat_file(1 : length(dat_file)-4), '_angle.mat');
elseif (amp_or_angle == 2)
    mat_file = strcat(dat_file(1 : length(dat_file)-4), '_complex.mat');
else
    disp("Invalid value for amp_or_angle, only accep 0 or 1.");
endif
save('-v6', mat_file, 'csi_array');

end
