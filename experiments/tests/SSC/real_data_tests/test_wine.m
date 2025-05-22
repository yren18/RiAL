load('wine.data');
tol = 1e-3;
r = 3;
kappa = 1;
lasso_constant = 1e-3;

opts_RiALSD.tol = tol;
opts_RiALSD.mxitr = 50000;
opts_RiALSD.lasso_constant = lasso_constant;
opts_RiALSD.sigma0 = 1e-1;
opts_RiALSD.rho = 1.05;
opts_RiALSD.tau1 = 0.999;
opts_RiALSD.tau2 = 0.9;
opts_RiALSD.subproblem_c1 = 1e-6;
opts_RiALSD.subproblem_c2 = 1e-4;
opts_RiALSD.inner_min_itr = 15;
opts_RiALSD.sparse_param = 1e-5;
opts_RiALSD.verbose = 0;

test_data = wine(:,1);
train_data = wine(:,2:end);
for i = 1:size(train_data,2)
train_data(:,i) = (train_data(:,i) - mean(train_data(:,i)))...
    ./(max(train_data(:,i)) - min(train_data(:,i)));
end
%% Parameter Setting
% lasso_constant = 1e-3;
opts.lasso_constant = lasso_constant;
entropy_penalty = 0;

%% Generate Laplacian
L = generate_laplacian_matrix(train_data,kappa);
%% Generate inital U
% r = 3;  
d = size(L,1);
%% Spectral Clustering
[nEigVec,~] = eigs(-L, r, 'largestreal');

U = nEigVec;


%% Algorithms parameters
opts.epsilon = 1e-3;
opts.mxitr = 1e4;
opts.R = lasso_constant*d;
opts.n = d*d;
opts.lambda = opts.epsilon/(2*opts.R);
opts.nbreakindex = 50;


%% Generate inital U
N = size(L,1);
[nEigVec,~] = eigs(-L, r, 'largestreal');
U = nEigVec;
opts.epsilon = tol;
opts.R = norm(lasso_constant*ones(N,N),'fro');
opts.n = N*N;
opts.mxitr = 1e4;
opts.nbreakindex = 50;
opts.lambda = opts.epsilon/(2*opts.R);

%% RADA-RGD
Y0 = ones(N,N);%
Y0 = Y0 + Y0';
Y0 = max(-lasso_constant,min(Y0,lasso_constant));

t_RADARGD = 0;
opts.adbeta = 1;
opts.nupdateU = 3;
opts.beta0 = 1;
opts.rho = 1.5;
opts.lasso_constant = lasso_constant;
t = tic;
[U_RADARGD, out_RADARGD]= RADA_RGD4SSCGr(U, opts, L, Y0);
t_RADARGD =t_RADARGD +toc(t);
NMI_RADA_RGD = nmi(kmeans(U_RADARGD,r),test_data);
fprintf('RADA_RGD & itr=%4d & f=%.8f & NMI=%.3f & time=%fs \n',out_RADARGD.itr, out_RADARGD.F_best, NMI_RADA_RGD, t_RADARGD);



%% RADA-PGD
opts.adbeta = 1;
opts.beta0 = 1;
opts.rho = 1.5;
t_RADAPGD = 0;
t = tic;
[U_RADAPGD, out_RADAPGD]= RADA_PGD4SSC(U, opts, L, Y0);
t_RADAPGD = t_RADAPGD + toc(t);
NMI_RADA_PGD = nmi(kmeans(U_RADAPGD,r),test_data);
fprintf('RADA_PGD & itr=%4d & f=%.8f & NMI=%.3f & time=%fs  \n',out_RADAPGD.itr, out_RADAPGD.F_best, NMI_RADA_PGD, t_RADAPGD);


%% RiAL-SD
% opts_RiALSD.tol = tol;
% opts_RiALSD.mxitr = 50000;
% opts_RiALSD.lasso_constant = lasso_constant;
% opts_RiALSD.sigma0 = 1e-4;
% opts_RiALSD.rho = 1.05;
% opts_RiALSD.tau1 = 0.999;
% opts_RiALSD.tau2 = 0.9;
% opts_RiALSD.subproblem_c1 = 1e-6;
% opts_RiALSD.subproblem_c2 = 1e-4;
% opts_RiALSD.inner_min_itr = 5;
% opts_RiALSD.sparse_param = 1e-5;
% opts_RiALSD.verbose = 0;

t_RiALSD = 0;
t = tic;
[U_RiAL_SD, out_RiALSD]= RiAL_SD4SSCGr(U, opts_RiALSD, L);
t_RiALSD = t_RiALSD + toc(t);
NMI_RiAL_SD = nmi(kmeans(U_RiAL_SD,r),test_data);
fprintf('RiAL_SD  & itr=%4d & f=%.8f & NMI=%.3f & time=%fs  \n',out_RiALSD.itr, out_RiALSD.F_best, NMI_RiAL_SD, t_RiALSD);
fprintf("\n");


