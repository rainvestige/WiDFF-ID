clear;
csi_trace = read_bf_file('/media/jyf/新加卷/learning/dataset/csi_data/2020-1-4_three_log.dat');

 
for i=1:length(csi_trace)%������ȡ����ݰ�ĸ���
        csi_entry = csi_trace{i};
        csi = get_scaled_csi(csi_entry); %��ȡcsi����  
        csi_all = abs(csi);
        csi =csi(1,:,:);
        csi1=abs(squeeze(csi).');          %��ȡ��ֵ(��ά+ת��)
        
        %ֻȡһ�����ߵ����
        first_ant_csi(:,i)=csi1(:,1);           %ֱ��ȡ��һ�����(����Ҫforѭ��ȡ)
        second_ant_csi(:,i)=csi1(:,2);
        third_ant_csi(:,i)=csi1(:,3);
        video(:,:,:,i)=csi_all;
                     
%         for j=1:30
%             csi15_end(i,:)=csi1(j,:);           %3���ŵ���j�����ز�
%         end
end
 
%����һ�����ߵ��ز�
%plot(first_ant_csi.')
plot(second_ant_csi.')
%plot(third_ant_csi.')
