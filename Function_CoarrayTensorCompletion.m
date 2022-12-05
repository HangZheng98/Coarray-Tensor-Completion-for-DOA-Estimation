function [err_theta, err_fe, err_all, theta_estimation, fe_estimation] = Function_CoarrayTensorCompletion(theta,fe,snapshot,SNR,Lx,Ly)

% written by Hang Zheng
% references: 
% H. Zheng, C. Zhou, A. L. F. de Almeida, Y. Gu, and Z. Shi 
% DOA estimation via coarray tensor completion with missing slices
% IEEE Int. Conf. Acoust., Speech, Signal Process. (ICASSP), Singapore, May 2022, pp. 5053–5057.

% H. Zheng, Z. Shi, C. Zhou, A. L. F. de Almeida, ad Y. Gu
% Coarray tensor completion for DOA estimation
% IEEE Trans. Aerosp. Electron. Syst.

% The tensor decompositon is implemented with tensorlab 3.0
% Reference
% Vervliet N., Debals O., Sorber L., Van Barel M. and De Lathauwer L. Tensorlab 3.0, Available online, Mar. 2016. URL: https://www.tensorlab.net/

% The tensor nucler norm-based tensor completion is implemented with the lrtc_tnn function
% Reference
% Canyi Lu, Jiashi Feng, Zhouchen Lin, Shuicheng Yan
% Exact Low Tubal Rank Tensor Recovery from Gaussian Measurements
% International Joint Conference on Artificial Intelligence (IJCAI). 2018

%% Parameters
derad = pi/180;
radeg = 180/pi;
twpi = 2*pi;

%% Setting of the coprime planar array
Nx = 3;
Ny = 4;
Mx = 2;
My = 3;
dd = 0.5;
d1=0:dd*Nx:(Mx*2-1)*Nx*dd;
d2=0:dd*Ny:(My*2-1)*Ny*dd;
d3=0:dd*Mx:(Nx-1)*Mx*dd;
d4=0:dd*My:(Ny-1)*My*dd;

%% Steering vector
iwave = size(theta, 2);   % number of sources
snr = SNR;                % input SNR (dB)
T = snapshot;             % number of snapshots
Amx=exp(-1i*twpi*d1.'*(sin(theta*derad).*cos(fe*derad)));   % steering matrix of the first sparse URA
Amy=exp(-1i*twpi*d2.'*(sin(theta*derad).*sin(fe*derad)));
Anx=exp(-1i*twpi*d3.'*(sin(theta*derad).*cos(fe*derad)));   % steering matrix of the second sparse URA
Any=exp(-1i*twpi*d4.'*(sin(theta*derad).*sin(fe*derad)));
sigma_n = 1;                          % noise power
sigma_s = sigma_n*(10.^(snr/10));     % desired signal power

%% Generate signal and noise
ss = [];
% for num_SS = 1:1           % this is the generation of coherent source, where s_{1} is randomly generated, and s_{k} is s_{1} multiplied with attenuation factor
%     s = randn(1,T) + 1i*randn(1,T);
%     s = s - mean(s);
%     ss(num_SS,:) = sqrt(sigma_s)*(s/sqrt(var(s)));
% end
% a = [1, randn(1,iwave-1) + 1i*randn(1,iwave-1)];
% ss = a.'*ss;

for num_SS = 1:iwave
    s = randn(1,T) + 1i*randn(1,T);
    s = s - mean(s);
    ss(num_SS,:) = sqrt(sigma_s)*(s/sqrt(var(s)));
end
nn1=[];       % Generate two independent noise
nn2=[];
for m = 1:4*Mx*My
    n = randn(1,T) + 1i*randn(1,T);
    n = n - mean(n);
    nn1(m,:) = sqrt(sigma_n)*(n/std(n));
end
for m = 1:Nx*Ny
    n = randn(1,T) + 1i*randn(1,T);
    n = n - mean(n);
    nn2(m,:) = sqrt(sigma_n)*(n/std(n));
end

%% Generate received signals of the two sparse URAs
X_1s=kr(Amx,Amy)*ss;
X_1=X_1s+nn1;

X_2s=kr(Anx,Any)*ss;
X_2=X_2s+nn2;

%% Model the received signals as sub-Nyquist tensors
X_1T = zeros(2*Mx,2*My,T);
X_2T = zeros(Nx,Ny,T);
for mx=1:2*Mx
    X_1T(mx,:,:)=X_1((mx-1)*2*My+1:My*2*mx,:);
end
for nx=1:Nx
    X_2T(nx,:,:)=X_2((nx-1)*Ny+1:Ny*nx,:);
end

%% Calculate the cross-correlation tensor
Cross_R=zeros(2*Mx,2*My,Nx,Ny);
for i=1:T
    P1=X_1T(:,:,i);
    P2=X_2T(:,:,i);
    Cross_R = Cross_R+ outprod(P1,conj(P2));
end
Cross_R = Cross_R/T;

%% Another way to formulate the cross-correlation tensor
% R=X_1*X_2'/T;
% R=R-eye(16,9);
% R_s_1=zeros(2*Mx,2*My,Nx);
% R_s_2=zeros(2*Mx,2*My,Nx,Ny);
% for i=1:Nx*Ny
%    for k=1:2*My
%      for t=1:2*Mx
%       R_s_1(t,k,i)=R((t-1)*2*Mx+k,i);
%      end
%    end
% end
% for i=1:Ny
%     for k=1:Nx
%     R_s_2(:,:,k,i)=R_s_1(:,:,(k-1)*Nx+i);
%     end
% end
% R1=R_s_2;

%% Reshape the cross-correlation tensor
R12=tens2mat(Cross_R,[1 3],[2 4]);

%% Arrange elements of the reshaped cross-correlation tensor to match the positions of virtual array
I1=[9 5 10 1 6 11 2 7 12 3 8 4];
I2=[19 13 20 7 14 21 1 8 15 22 2 9 16 23 3 10 17 24 4 11 18 5 12 6];
R12_r=R12(I1,I2);

%% Add zeros rows and colums corresponding to the holes of the virtual array
R12_r=[R12_r(1,:);zeros(1,size(R12_r,2));R12_r(2:end,:)];
R12_r=[R12_r(1:12,:);zeros(1,size(R12_r,2));R12_r(13:end,:)];
R12_r=[R12_r(:,1),zeros(size(R12_r,1),2),R12_r(:,2:end)];
R12_r=[R12_r(:,1:25),zeros(size(R12_r,1),2),R12_r(:,26:end)];
R12_r=[R12_r(:,1:5),zeros(size(R12_r,1),1),R12_r(:,6:end)];
R12_r=[R12_r(:,1:24),zeros(size(R12_r,1),1),R12_r(:,25:end)];

%% Extend the mirror part of the virtual array signal
R12_m=R12_r;
R12_mirror=conj(R12_m);
R12_mirror=flipud(R12_mirror);
R12_mirror=fliplr(R12_mirror);

%% Formulate the 3-D coarray tensor
U = cat(3,R12_r,R12_mirror);
p1 = size(U,1);
p2 = size(U,2);

%% Formulate the 5-D dimensional increament tensor
Q1 = Lx;
Q2 = Ly;
Z = zeros(Q1,Q2,2,p1+1-Q1,p2+1-Q2);
for s_h = 1:p2+1-Q2
    Z1 = zeros(Q1,Q2,2,p1+1-Q1);
    for s_v = 1:p1+1-Q1
        Z1(:,:,:,s_v) = U(s_v:s_v+Q1-1,s_h:s_h+Q2-1,:);
    end
    Z(:,:,:,:,s_h) = Z1;
end

%% Reshape the 5-D tensor to a 3-D structured on
V=desegmentize(Z,'dims',[1 2]);
V=desegmentize(V,'dims',[3 4]);
V_r = permute(V,[1 3 2]);

%% finding the optimal sub-coarray tensor size by comparsing the DPR of missing elements
% distance = [];
% for q_h = 2:13
%     for q_v = 2:29
% Q1 = 7;
% Q2 = 14;
% Q1 = q_h;
% Q2 = q_v;
% locat = find(V_r == 0);
% num_zero = size(locat,1);
% x = fix((locat-1)/(size(V_r,1)*size(V_r,2)))+1;
%
% y = fix(rem((locat-1),(size(V_r,1)*size(V_r,2)))/size(V_r,1)) + 1;
%
% z = rem(rem((locat-1),(size(V_r,1)*size(V_r,2))),size(V_r,1))+1;
%
% dis = 0;
% lo = 0;
% for h =1 : size(x)-1
%     for t = h+1:size(x)
%     mis = sqrt((x(h) - x(t))^(2) + (y(h) - y(t))^(2) + (z(h) - z(t))^(2));
%     dis = dis+ mis;
%     lo = lo+1;
%     end
% end
% % for h =1 : size(x)
% %     for t = 1:size(x)
% %         if ((x(h) - x(t)) + (y(h) - y(t)) + (z(h) - z(t))) == 1
% % %     mis = sqrt((x(h) - x(t))^(2) + (y(h) - y(t))^(2) + (z(h) - z(t))^(2));
% % %     dis = dis+ mis;
% % %           lo = lo+1;
% %         end
% %     end
% % end
% %lo = lo/2;
% % dis = dis/((size(V_r,1)*size(V_r,2)));
% % dis = dis/num_zero;
% rate = num_zero/(size(V_r,1)*size(V_r,2)*2);
% distance = [distance; Q1, Q2, num_zero, dis, rate];
%     end
% end
%
% distance_sort  = sortrows(distance,-6);

%% Complete the strutured coarray tensor with low-rank regularization
omega = find(V_r~=0);   % a mask tensor to indicate the observed elements
opts.DEBUG = 0;
rho = 1e-4;
opts.mu = rho;
[V_comple,obj,out_15]= lrtc_tnn(V_r,omega,opts);  % low-rank coarray tensor completion with nuclear norm minimizatio
%semilogy(out_15(:,1), out_15(:,2), 'r-', 'LineWidth', 3,'MarkerSize', 13);

%RSE = norm(V_comple - V_r, 'fro')/ norm(V_r, 'fro');
%% Decompose the completed coarray tensor to obtain CP factors
options.Initialization=@cpd_gevd;        % initilization
options.Compression=0;
options.Algorithm=@cpd_als;              % using the alternative least square optimization
options.AlgorithmOptions.LineSearch=@cpd_els;
options.AlgorithmOptions.TolFun=1e-20;
options.AlgorithmOptions.TolX=1e-20;
L=cpd(V_comple,iwave,options);

P_1 = p1+1-Q1;
P_2 = p2+1-Q2;

%% Retreive directional parameters \mu and \nu from the CP factors with scale normalization
L11 = cell2mat(L(1));
L22 = cell2mat(L(2));
D1=[];
t1=0;
t2=0;
for i=1:(Q1*Q2-1)
    if mod(i,Q1)~=0
        t1=t1+1;
        D1=[D1; angle(L11(i+1,:)./L11(i,:))/(pi)];
    end
end
for i=1:(P_1*P_2-1)
    if mod(i,P_1)~=0
        t1=t1+1;
        D1= [D1; angle(L22(i+1,:)./L22(i,:))/(pi)];
    end
end
D2=[];
for i=(Q1+1):(Q1*Q2)
    t2=t2+1;
    D2= [D2; angle(L11(i,:)./L11(i-Q1,:))/(pi)];
end
for i=(P_1+1):(P_1*P_2)
    t2=t2+1;
    D2= [D2; angle(L22(i,:)./L22(i-P_1,:))/(pi)];
end
D1(all(isnan(D1),2),:) = [];
D2(all(isnan(D2),2),:) = [];
m1 = median(D1,1);
m2 = median(D2,1);

for p = 1:iwave
    for l = 1:size(D1,1)
        if abs(D1(l,p)-m1(p)) >= 0.045
            D1(l,p) = 0;
        end
    end
    for l =1:size(D2,1)
        if abs(D2(l,p)-m2(p)) >= 0.045
            D2(l,p) = 0;
        end
    end
end

D1_f = zeros(1,iwave);
D2_f = zeros(1,iwave);

for k = 1:iwave
    D1_f(k) = sum(D1(:,k))/(sum(sum(D1(:,k)~=0)));
    D2_f(k) = sum(D2(:,k))/(sum(sum(D2(:,k)~=0)));
end

%% 2-D DOA estimation
theta_estimation=asin(sqrt(D1_f.^2+D2_f.^2))*radeg;
fe_estimation=atan(D2_f./D1_f)*radeg;

% theta_estimation = abs(theta_estimation);
% fe_estimation = abs(fe_estimation);
err_theta = norm(theta_estimation - theta)^(2);
err_fe = norm(fe_estimation - fe)^(2);
err_all = norm(theta_estimation - theta)^(2) + norm(fe_estimation - fe)^(2);
end