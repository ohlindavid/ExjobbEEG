function sig = reass2sig(t_vec,f_vec,real_sig,Fs,N,M,c)
% Find peaks:
    t_vec = round(t_vec*Fs);
    f_vec = f_vec;
    time=[-M/2+1:M/2]';
    H=exp(-((time).^2)/(2*c^2))/sqrt(sqrt(pi))/sqrt(c);
    Hmat=zeros(N,length(t_vec));

% Create Cosine and Sine:
    for i=1:length(t_vec)
        nvect=[t_vec(i)-M/2:t_vec(i)+M/2-1]';
        nvect = nvect.*(nvect>0) + (nvect<=0);
        nvect(nvect>N) = N;
        Hmat(nvect,i)=H;
        COSMAT(:,i)=cos(2*pi*f_vec(i).*[0:N-1]'/Fs).*Hmat(:,i);
        SINMAT(:,i)=sin(2*pi*f_vec(i).*[0:N-1]'/Fs).*Hmat(:,i);
    end
    
    X = [COSMAT SINMAT];
% Solve system:
    theta = pinv(X)*real_sig;
% Generate Signal and return:
    sig = X*theta;
end