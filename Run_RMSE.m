clear;
clc;

%% Define the elevation and azimuth
theta = [30.6 40.5];
fe = [25.6 50.5] ;
Lx = 7;
Ly = 14;

RMSE_CPA_TensorCompletion_theta_SNR = [0 0 0 0 0 0 0];
RMSE_CPA_TensorCompletion_fe_SNR = [0 0 0 0 0 0 0];
RMSE_CPA_TensorCompletion_theta_SNAP = [0 0 0 0 0 0 0];
RMSE_CPA_TensorCompletion_fe_SNAP = [0 0 0 0 0 0 0];
Trial = 1000;

%% RMSE with different SNR and T = 300
snapshot = 300;
SNR = [-15:5:15];
for SNR_T = 1:size(SNR, 2)
    for tr = 1:Trial
        tic
        [err_theta, err_fe, err_all, theta_estimation, fe_estimation] = Function_CoarrayTensorCompletion(theta,fe,snapshot,SNR(SNR_T),Lx,Ly);
        RMSE_CPA_TensorCompletion_theta_SNR(SNR_T) = RMSE_CPA_TensorCompletion_theta_SNR(SNR_T) +  err_theta;
        RMSE_CPA_TensorCompletion_fe_SNR(SNR_T) = RMSE_CPA_TensorCompletion_fe_SNR(SNR_T) +  err_fe;
        message = ['Proposed-SNR', ' SNR = ', num2str(SNR(SNR_T)) , ' Trial = ' , num2str(tr), ' Error = ' , num2str(err_all), ' Time = ', num2str(toc)];
        disp(message)
    end
end
RMSE_CPA_TensorCompletion_theta_SNR = sqrt(RMSE_CPA_TensorCompletion_theta_SNR/(Trial));
RMSE_CPA_TensorCompletion_fe_SNR = sqrt(RMSE_CPA_TensorCompletion_fe_SNR/(Trial));

%% RMSE with differet number of snapshots and SNR = 0dB
snapshot = [36 100:100:600];
SNR = 0;
for Snap_T = 1:size(snapshot, 2)
    for tr = 1:Trial
        tic
        [err_theta, err_fe, err_all, theta_estimation, fe_estimation] = Function_CoarrayTensorCompletion(theta,fe,snapshot(Snap_T),SNR, Lx, Ly);
        RMSE_CPA_TensorCompletion_theta_SNAP(Snap_T) = RMSE_CPA_TensorCompletion_theta_SNAP(Snap_T) +  err_theta;
        RMSE_CPA_TensorCompletion_fe_SNAP(Snap_T) = RMSE_CPA_TensorCompletion_fe_SNAP(Snap_T) +  err_fe;
        message = ['Proposed-Snapshot', ' Snapshot = ', num2str(snapshot(Snap_T)) , ' Trial = ' , num2str(tr), ' Error = ' , num2str(err_all),  ' Time = ', num2str(toc)];
        disp(message)
    end
end
RMSE_CPA_TensorCompletion_theta_SNAP = sqrt(RMSE_CPA_TensorCompletion_theta_SNAP/(Trial));
RMSE_CPA_TensorCompletion_fe_SNAP = sqrt(RMSE_CPA_TensorCompletion_fe_SNAP/(Trial));