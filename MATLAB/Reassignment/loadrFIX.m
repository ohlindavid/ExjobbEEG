clear;
basic_path = "C:\Users\Kioskar\Desktop\Testing exjobb\Albin_Damir\all_subj_crop\";
subjects = ls(basic_path + "Morlet\");
subjects = string(subjects);
subjects = subjects(3:end);

for subj_nr = 1:length(subjects)
    subj = subjects(subj_nr);
    NSTEP=1;
    downsample = 4;
    Fs=512/downsample; % ursprunglig samplingsfrekvens: 512 Hz
    NFFT=512;
    nch = 31;
    lambda = 3; % Window parameter
    candsigvect = [ lambda ]; % Candidate sigma for the unknown sigma of X
    resize = 4;
    H = [];
    trial = dir(basic_path + "Morlet\" + subj +"\");
    
    for k = 3:(length(trial))
        
        load(basic_path + "Morlet\" + subj + "\" +trial(k).name)
        trial(k).name
        processed = [];
        processeds = processed;
        processedsig = [];
        processedsiggau = [];
        
        for ch = 1:nch
            
            temp = eval(trial(k).name);
            startt = 1;
            endd = length(temp);
            X = decimate(temp(startt:endd,ch),downsample);

            for i=1:length(candsigvect)
                
                candsig = candsigvect(i);
                
                [SS,MSS,TI,FI,H]=screassignspectrogram1(real(X),lambda,candsig,NFFT,NSTEP,Fs);
                
                processed(ch,i,:,:) = MSS;
                processeds(ch,i,:,:) = SS;
                min_dist = 5;
                components_keep = 25;
                scale = 10;
                
                rnew = [];
                cnew = [];
                pksnew = [];
                locs1 = find(imregionalmax(SS));
                pks = SS(locs1);
                [r,c] = ind2sub(size(SS), locs1);
                [pks,order] = sort(pks,'descend');
                r = r(order);
                c = c(order);
                idx = 1;
                while ~isempty(r)
                    coord = [r(1),c(1)];
                    all_coords = [r,c];
                    close_coords = vecnorm(all_coords-coord,2,2)<min_dist;
                    rnew(idx) = r(1);
                    cnew(idx) = c(1);
                    pksnew(idx) = pks(1);
                    idx = idx + 1;
                    pks(close_coords) = [];
                    r(close_coords) = [];
                    c(close_coords) = [];
                end
                
                [maxpks,idxs]  = maxk(pksnew,components_keep);
                
                maxr = rnew(idxs);
                maxc = cnew(idxs);
                
                t_vec = maxc/128;
                f_vec = maxr/4;
                M = lambda*10;
                N = length(X);
                back2sigCorrect = reass2sig(t_vec,f_vec,X,Fs,N,M,candsig);
                back2sigCorrectA = reass2sig_only_A(t_vec,f_vec,X,Fs,N,M,candsig)
                [back2sigVanilla,T] = multigaussdata(N,(M)*ones(length(t_vec),1)',diag(SS(maxr,maxc))'./max(diag(SS(maxr,maxc))),t_vec,f_vec,zeros(length(t_vec),1)',Fs);
                
                
                
                
                rnew = [];
                cnew = [];
                pksnew = [];
                locs = find(imregionalmax(MSS));
                pks = MSS(locs);
                [r,c] = ind2sub(size(MSS), locs);
                [pks,order] = sort(pks,'descend');
                r = r(order);
                c = c(order);
                jdx = 1;
                while ~isempty(r)
                    coord = [r(1),c(1)];
                    all_coords = [r,c];
                    close_coords = vecnorm(all_coords-coord,2,2)<min_dist;
                    rnew(jdx) = r(1);
                    cnew(jdx) = c(1);
                    pksnew(jdx) = pks(1);
                    jdx = jdx + 1;
                    pks(close_coords) = [];
                    r(close_coords) = [];
                    c(close_coords) = [];
                end
                
                [maxpks,jdxs]  = maxk(pksnew,components_keep);
                maxr = rnew(jdxs);
                maxc = cnew(jdxs);
                
                t_vec = maxc/128;
                f_vec = maxr/4;
                N = length(X);
                M = lambda*10;
                back2sigCorrectR = reass2sig(t_vec,f_vec,X,Fs,N,M,candsig);
                back2sigCorrectRA = reass2sig_only_A(t_vec,f_vec,X,Fs,N,M,candsig)
                [back2sigVanillaR,T] = multigaussdata(N,(M)*ones(length(t_vec),1)',diag(MSS(maxr,maxc))'./max(diag(MSS(maxr,maxc))),t_vec,f_vec,zeros(length(t_vec),1)',Fs);
                sigVanilla(ch,:) = back2sigVanilla;
                sigCorrect(ch,:) = back2sigCorrect;
                sigVanillaR(ch,:) = back2sigVanillaR;
                sigCorrectR(ch,:) = back2sigCorrectR;
                
                sigCorrectA(ch,:) = back2sigCorrectA;
                sigCorrectRA(ch,:) = back2sigCorrectRA;
            end
        end
        %dlmwrite(basic_path + "SpectrogramRecCorrect\" + subj + "\" + trial(k).name,sigCorrect');
        %dlmwrite(basic_path + "SpectrogramRec\" + subj + "\"  + trial(k).name,sigVanilla');
        %dlmwrite(basic_path + "ReassignmentRecCorrect\" + subj + "\" + trial(k).name,sigCorrectR');
        %dlmwrite(basic_path + "ReassignmentRec\" + subj + "\" + trial(k).name,sigVanillaR');
        %dlmwrite(basic_path + "Reassignment2D\" + subj + "\" + trial(k).name,processed);
        %dlmwrite(basic_path + "Spectrogram2D\" + subj + "\" + trial(k).name,processeds);
        
        dlmwrite(basic_path + "SpectrogramRecCorrectA\" + subj + "\" + trial(k).name,sigCorrectA');
        dlmwrite(basic_path + "ReassignmentRecCorrectA\" + subj + "\" + trial(k).name,sigCorrectRA');

        clear(trial(k).name);
    end
end