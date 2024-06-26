clear all;clc;
% close all;
gt_x=2:251;
gt_y=zeros(1,250)-80;

% load matched_filter_data
% load HighSNR_Target_Mask_20dBWeakerTarget_Speed_1mPerS;gt_y([42,77,87])=[-1.5,-20,0];
% load HighSNR_Target_Mask_20dBWeakerTarget_Speed_10mPerS;gt_y([35,80,90])=[-1.5,-20,0];
% load HighSNR_Target_Mask_20dBWeakerTarget_Speed_300mPerS;gt_y([35,80,90])=[-1,-20,0];%8
% load HighSNR_Target_Mask_20dBWeakerTarget_Speed_600mPerS;gt_y([35,80,90])=[-1,-20,0];%5
% load HighSNR_Target_Mask_30dBWeakerTarget_Speed_1mPerS;gt_y([42,80,90])=[-1.5,-29,0];
% load HighSNR_Target_Mask_30dBWeakerTarget_Speed_10mPerS;gt_y([35,80,90])=[-1.5,-29,0];
% load HighSNR_Target_Mask_30dBWeakerTarget_Speed_300mPerS;gt_y([35,80,90])=[-1.5,-29,0];%5
% load HighSNR_Target_Mask_30dBWeakerTarget_Speed_600mPerS;gt_y([35,80,90])=[-1.5,-29,0];%5
% load HighSNR_Target_Mask_50dBWeakerTarget_Speed_1mPerS;gt_y([42,80,90])=[-1.5,-49,0];
load HighSNR_Target_Mask_50dBWeakerTarget_Speed_10mPerS;gt_y([35,80,90])=[-1.5,-49,0];%
 % load HighSNR_Target_Mask_50dBWeakerTarget_Speed_300mPerS;gt_y([35,80,90])=[-1.5,-49,0];
% load HighSNR_Target_Mask_50dBWeakerTarget_Speed_600mPerS;gt_y([35,80,90])=[-1.5,-49,0];%2
% FF AA FF AA
% load HighSNR_Target_Mask_20dBWeakerTarget_Speed_1mPerS_100;
% load HighSNR_Target_Mask_20dBWeakerTarget_Speed_10mPerS_100;
% load HighSNR_Target_Mask_20dBWeakerTarget_Speed_300mPerS_100;
% load HighSNR_Target_Mask_20dBWeakerTarget_Speed_600mPerS_100;
% load HighSNR_Target_Mask_30dBWeakerTarget_Speed_10mPerS_100;
% load HighSNR_Target_Mask_30dBWeakerTarget_Speed_300mPerS_100;
% load HighSNR_Target_Mask_30dBWeakerTarget_Speed_600mPerS_100;
% load HighSNR_Target_Mask_50dBWeakerTarget_Speed_10mPerS_100;
% load HighSNR_Target_Mask_50dBWeakerTarget_Speed_300mPerS_100;
% load HighSNR_Target_Mask_50dBWeakerTarget_Speed_600mPerS_100;


offset=1
N = 13; % Number of WaveSample
n = (0:(N-1)).';  % WaveSample Index
L = 100-N; %gt_x=gt_x-2;   % Length of Processing Window
% L = 250-N;    % Length of Processing Window
noise_db = -70; % Noise Level Power

% alpha = [1.9 1.7 1.6 1.4]; % Control the Speed of Convergance

alpha = [1.9 1.8]; % Control the Speed of Convergance


s = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1 ,1].';

S_MF = zeros(L+3*(N-1),L+2*(N-1)); % Henkel Matrix whit Shifted 's' for Match Filter
s_MF = [s;zeros(2*(N-1)+L-1,1)]; % Waveform Zeropadded for Henkel Matrix 'S'
% Henkel Matrix 's' Preparation
S_MF(:,1) = s_MF;
for i=1:L-1+2*(N-1)
    s_MF = [0;s_MF(1:end-1)];
    S_MF(:,i+1) = s_MF;
end

for l=1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%s = exp(1i*pi/N*n.^2); % Waveform

targets = [-5,50,55,53,103]; % targets locations
rcs = [1,1,1,.004,1];
%doppler_deg = zeros(length(targets),1); % Doppler Phase Shift in Degrees
doppler_deg = [10,-15,18,-10,-20];    % Doppler Phase Shift in Degrees

%[y_noisy , G, sigma] = profile_builder(s, targets, doppler_deg, rcs, L, N, noise_db);
sigma = 1e-5;
%y_noisy = [(mf_data_in_I(l,:) + 1i*mf_data_in_Q(l,:)),zeros(1,N-1)].'/2^10;
y_noisy  = ((mf_data_in_I(l,offset:offset+98) + 1i*mf_data_in_Q(l,offset:offset+98))).'/2^10;
y_mf_out = sqrt(mf_data_out_I(l,offset:offset+98).^2 + mf_data_out_I(l,offset:offset+98).^2);

figure; plot(0:1:98, mf_data_in_I(l,offset:offset+98),'b',0:1:98, mf_data_in_Q(l,offset:offset+98),':r','LineWidth',2); legend('I','Q');xlabel 'Range cell index';xlim([0 150]);
% figure; plot(20*log10(abs(y_mf_out/max(abs(y_mf_out)))));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;

plot(gt_x,gt_y,'-^','Color','k','LineWidth',1.7),hold on,axis tight; 

rho = S_MF(1:end-N+1,:)'*[zeros(N-1,1);y_noisy];   % MF Operation
plot(-N+1:L+N-2,20*log10(abs(rho/max(abs(rho)))),':','Color','r','LineWidth',1.5),hold on,axis tight 
% plot(-N+1:L+N-2,20*log10(abs(G)),'k'),hold on,axis tight 

X = general_APC(s, y_noisy, N, alpha, sigma);
% X = DC_APC(s, y_noisy, alpha, sigma);
% X2 =  general_APC2(s, y_noisy, N, alpha, sigma);
% plot(1:141,20*log10(abs(X/max(abs(X)))),'-','Color','b','LineWidth',.5);
%plot(0:236,20*log10(abs(X/max(abs(X)))),'-','Color','b','LineWidth',.5);
plot(0:size(X,2)-1,20*log10(abs(X/max(abs(X)))),'-','Color','b','LineWidth',.5);

% X = APC_ER(s, y_noisy, N, L, alpha, sigma);
% plot(-N+1:L-1+N-1,20*log10(abs(X/max(abs(X)))),'--','Color','k','LineWidth',1.5)
% 
% X = DC_APC(s, y_noisy, N, alpha, sigma);
% plot(0:L-1,20*log10(abs(X/max(abs(X)))),':','Color','y','LineWidth',1.5)
% 
% X = DC_APC(s, y_noisy, N, alpha, sigma);
% plot(0:L-1,20*log10(abs(X/max(abs(X)))),'--','Color','R','LineWidth',1.5)
% 
% X = MF_RMMSE(s, y_noisy, N, L, 4, 4, alpha, sigma, -60);
% plot(0:L-1,20*log10(abs(X/max(abs(X)))),':','Color','c','LineWidth',1.5)

% Plot result
plot(0*ones(1,length(0:-1:noise_db-10)),0:-1:noise_db-10,'--','Color','R','LineWidth',2)
plot((150)*ones(1,length(0:-1:noise_db-10)),0:-1:noise_db-10,'--','Color','R','LineWidth',2)
%plot(20*log10(abs(X/max(X))),'--','Color','G','LineWidth',1.5)
% % axis([-Inf,Inf,noise_db-10,0]);
axis([0,150,noise_db-10,0]);
%legend 'MF' 'Ground Truth' 'APC' 'APC-ER' 'CMT-APC' 'DC-APC' 'MF-RMMSE'
legend 'GT' 'MF' 'APC';% 'DC-APC''APC-ER' 'CMT-APC' 'DC-APC' 'MF-RMMSE'
hold on
grid on
xlabel 'Range cell index'
ylabel 'Power(dB)'
pause(.2);
hold off
% figure; plot(0:L-1,10*log10(abs(X2/max(abs(X2)))),'-','Color','b','LineWidth',.5);hold on;
% plot(0:L-1,20*log10(abs(X/max(abs(X)))),'-','Color','r','LineWidth',.5);

end