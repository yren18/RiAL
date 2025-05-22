%% feature engineering
load('letter_data2.mat')
letterdata2 = letterdata2(1:1000,:);
letterdata2 = letterdata2(1:1000,:);
test_data = letterdata2(:,1);
train_data = letterdata2(:,2:end);
for i = 1:size(train_data,2)
train_data(:,i) = (train_data(:,i) - mean(train_data(:,i)))...
    ./(max(train_data(:,i)) - min(train_data(:,i)));
end

%% Parameter Setting
lasso_constant = 1e-3;
opts.lasso_constant = lasso_constant;
entropy_penalty = 0;

%% Generate Laplacian
L = generate_laplacian_matrix(train_data, 1);
% L = L;
%% Generate inital U
r = 17;  d = size(L,1);
%% Spectral Clustering
[nEigVec,~] = eigs(-L, r, 'largestreal');
U = nEigVec;

%% Algorithms parameters
opts.epsilon = 1e-3;
opts.mxitr = 5e4;
opts.R = lasso_constant*d;
opts.n = d*d;
opts.lambda = opts.epsilon/(2*opts.R);
opts.nbreakindex = 50;

%% ADMM Result
% opts.lambda0 = 0.01;
% opts.sigma0 = 0.1;
% t_NADMM = 0;
% t = tic;
% [U_ADMM, ~, out_NADMM] = NADMM(U, L, lasso_constant, opts);
% t_NADMM = t_NADMM + toc(t); 
% % fprintf(fileID,' ADMM & %4d & %.8f & %fs & %e \\\\ \n',...
% %     out_ADMM.itr, function_multi_val_ADMM, t11, opts.sigma);
% fprintf(' NADMM & itr=%4d & f=%.8f & time=%fs \n', out_NADMM.itr, out_NADMM.F_best, t_NADMM);
% Score_ADMM = nmi(kmeans(U_ADMM,r),test_data);


%% ARPGDA
Y0 = ones(d,d);%draw y0 randomly
Y0 = Y0 + Y0';
Y0 = max(-lasso_constant,min(Y0,lasso_constant));
opts.beta0 = 1e-9; opts.rho = 1.5;
% t_ARPGDA = 0;
% t = tic;
% [U_ARPGDA, out_ARPGDA]= ARPGDA4SSCGr(U, opts, L, Y0);
% t_ARPGDA =t_ARPGDA +toc(t);
% fprintf('ARPGDA & itr=%4d & f=%.8f & time=%fs \n',out_ARPGDA.itr, out_ARPGDA.F_best, t_ARPGDA);
% Score_ARGbb = nmi(kmeans(U_ARPGDA,r),test_data);


%% RADARGD
t_RADARGD = 0;
opts.adbeta = 1;
opts.nupdateU = 3;
opts.beta0 = 1;
opts.rho = 1.5;
t = tic;
[U_RADARGD, out_RADARGD]= RADA_RGD4SSCGr(U, opts, L, Y0);
t_RADARGD =t_RADARGD +toc(t);
fprintf('RADA_RGD & itr=%4d & f=%.8f & time=%fs \n',out_RADARGD.itr, out_RADARGD.F_best, t_RADARGD);
% Score_RADA_RGD = nmi(kmeans(U_RADARGD,r),test_data);


%% RADAPGD
muset = 1e-4; 
rhoset = 1.5;
opts.adbeta = 1;
opts.beta0 = 1;
opts.rho = 1.5;
t_RADAPGD = 0;
t = tic;
[U_RADAPGD, out_RADAPGD]= RADA_PGD4SSC(U, opts, L, Y0);
t_RADAPGD = t_RADAPGD + toc(t);
fprintf('RADAPGD & itr=%4d & f=%.8f & time=%fs  \n',out_RADAPGD.itr, out_RADAPGD.F_best, t_RADAPGD);
Score_RADA_PGD = nmi(kmeans(U_RADAPGD,r),test_data);

% format longg


%% DSGM 
% t_DSGM = 0;
% opts.lambda0 = 0.01;
% t = tic;
% [U_DSGM, out_DSGM]= DSGM4SSCGr(U, opts, L, Y0);
% t_DSGM =t_DSGM +toc(t);
% fprintf('DSGM & itr=%4d & f=%.8f & time=%fs \n',out_DSGM.itr, out_DSGM.F_best, t_DSGM);
% Score_DSGM = nmi(kmeans(U_DSGM,r),test_data);


%% RiAL_SD
opts_RiALSD.tol = opts.epsilon;
opts_RiALSD.mxitr = 50000;
opts_RiALSD.lasso_constant = lasso_constant;
opts_RiALSD.sigma0 = 1e-3;
opts_RiALSD.rho = 1.5;
opts_RiALSD.tau1 = 0.9;
opts_RiALSD.tau2 = 0.5;
opts_RiALSD.subproblem_c1 = 1e-6;
opts_RiALSD.subproblem_c2 = 1e-4;
opts_RiALSD.inner_min_itr = sqrt(d)/3;
opts_RiALSD.sparse_param = 1e-5;
opts_RiALSD.verbose = 0;

t_RiALSD = 0;
t = tic;
[U_RiAL_SD, out_RiALSD]= RiAL_SD4SSCGr(U, opts_RiALSD, L);
t_RiALSD = t_RiALSD + toc(t);
% Score_RiAL_SD = nmi(kmeans(U_RiAL_SD,r),test_data);
fprintf('RiAL_SD  & itr=%4d & f=%.8f & NMI=%.3f & time=%fs  \n',out_RiALSD.itr, out_RiALSD.F_best, Score_RiAL_SD, t_RiALSD);

rng(123);
Score_RADA_RGD = 0;
Score_RADA_PGD = 0;
Score_RiAL_SD = 0;
nrun = 50;
for i = 1:nrun    
    Score_RADA_RGD = Score_RADA_RGD + nmi(kmeans(U_RADARGD,r),test_data)/nrun;
    Score_RADA_PGD = Score_RADA_PGD + nmi(kmeans(U_RADAPGD,r),test_data)/nrun;
    Score_RiAL_SD = Score_RiAL_SD + nmi(kmeans(U_RiAL_SD,r),test_data)/nrun;
end


fprintf("\n")
fprintf('NMI & $%.4f$ & $%.4f$ & $%.4f$ \\\\ \n',...
    Score_RADA_PGD, Score_RADA_RGD, Score_RiAL_SD);
fprintf('$\\Phi$ & $%.4f$ & $%.4f$ & $%.4f$ \\\\ \n',...
    out_RADAPGD.F_best, out_RADARGD.F_best, out_RiALSD.F_best);
fprintf('iter & $%.0f$ & $%.0f$ & $%.0f$ \\\\ \n',...
    out_RADAPGD.itr, out_RADARGD.itr, out_RiALSD.itr);
fprintf('cpu & $%.2f$ & $%.2f$ & $%.2f$ \\\\ \n',...
    t_RADAPGD, t_RADARGD, t_RiALSD);



