clear;
close all;
clc;
E = 1;
time = 4;
Fs = 1024/2;
N = 512;
M = N/2;
noreal = 10;
folder = "C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Simulated\5comp_1ch\";
folder_noise= "C:\Users\Oskar\Documents\GitHub\exjobb\Testing Sets\Simulated\5comp_1ch_noise\";

for j=1:60

    % Placeholder matrix where element (i,j) notes the distance matrix between
    % channel i and j. 
    dMatrix = rand(E,E);
    dMatrix = dMatrix/norm(dMatrix);
    dMatrix = (dMatrix'+dMatrix)/2;

    % Noise weight vector. [alpha , 1/f , m]
    w_noise = [1 , 1 , 1]';
    w_noise = w_noise/norm(w_noise);

    % Simulate alpha noise
    [B,A] = butter(3,[8/Fs*2 12/Fs*2]);
    alpha_power = abs(freqz(B,A,M)).^2;
    alpha_power = alpha_power./sum(alpha_power)/2*N;
    alpha_noise = zeros(N,E);
    alpha_wn1 = randn(N,E);
    for i=1:E
       alpha_noise(:,i)=filter(B,A,alpha_wn1(:,i));
       alpha_noise(:,i)=alpha_noise(:,i)./sqrt(alpha_noise(:,i)'*alpha_noise(:,i))*sqrt(N); % Normalized to power one
    end
    
    % Simulate pink noise
    x = pinknoise(N,E)*20;
    y = fft(x);
    n = length(x);          % number of samples
    f = (0:n-1)*(Fs/n);     % frequency range
    power = abs(y).^2/n;    % power of the DFT
    oof_noise  = x;

    % Simulate measurement noise
    measurement_noise = randn(N,E);

    % Define and calc Coherence between channels of different noises.
    Coh_oof = exp(-0.01*(dMatrix.^2));
    Coh_alpha = exp(-0.002*(dMatrix.^2));

    tot_noise = w_noise(1)*alpha_noise + w_noise(2)*measurement_noise + w_noise(3)*oof_noise;
   
   for  i=1:1
    [X,T] = multigaussdata(N,[100 100 300 512 10^7],[0.5 0.5 1 1 1],[0.7 0.1 0.2 0.5 0],[6 6 12 17 2],[0 0.05 0 0 0],Fs);
    [X2,T2] = multigaussdata(N,[100 100 300 512 10^7],[0.5 0.5 1 1 1],[0.7 0.1 0.2 0.5 0],[6 6 0 17 2],[0 0.05 0 0 0],Fs);
    %[X2,T2] = multigaussdata(N,[1012 100],[1 0.1],[1 1],[2 32],[0 0],Fs);
    %[X,T] = multigaussdata(N,[512, 100, 200, 700, 50],[1/2,1/2,1/2,1/2],[1,1,0.5,2],[4,12,8,17],rand(noreal,4)*2*pi,Fs);
    %[X2,T2]= multigaussdata(N,[512,60,150,650],[1/2,1/2,1/2,1/2],[1,0.5,2,0.2],[5,4,17,13],rand(noreal,4)*2*pi,Fs);
    Ain = X;
    Bin = X2;
   end
   
   SNR=1.1;
   lambda=10^(SNR/10);
   lambda2 = 10^(SNR/10);
   Aout(:,1) = sqrt(lambda/(lambda+1)).*Ain(:,1)./norm(Ain(:,1))+1/sqrt(lambda+1)*tot_noise(:,5)/norm(tot_noise(:,5));
   Aout(:,2) = sqrt(lambda/(lambda+1)).*Ain(:,2)./norm(Ain(:,2))+1/sqrt(lambda+1)*tot_noise(:,6)/norm(tot_noise(:,6));
   Bout(:,1) = sqrt(lambda2/(lambda2+1)).*Bin(:,1)./norm(Bin(:,1))+1/sqrt(lambda2+1)*tot_noise(:,5)/norm(tot_noise(:,5));
   Bout(:,2) = sqrt(lambda2/(lambda2+1)).*Bin(:,2)./norm(Bin(:,2))+1/sqrt(lambda2+1)*tot_noise(:,6)/norm(tot_noise(:,6));
   csvwrite(folder + 'A' + string(j),Ain);
   csvwrite(folder + 'B' + string(j),Bin);
   csvwrite(folder_noise + 'A' + string(j),Aout);
   csvwrite(folder_noise + 'B' + string(j),Bout);
end
plot(Aout)
hold on;
plot(Bout)
