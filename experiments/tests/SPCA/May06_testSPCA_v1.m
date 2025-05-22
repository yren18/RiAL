%% small and medium size SPCA test

% nrun = 20;
rng(123)

% nrun = 1;
% d = 5000;
% r = 50;
% mu = 0.5;
% sigma0 = 1.5; 
% sigma0_SD = 1.0e-3;
% sparse_param = 1.0e-5;
% 
% [Result1, out1] = RunRep(d, 50, r, mu, 1, nrun, sigma0, sigma0_SD, sparse_param, 10, 5000);


nrun = 10;
d = 2000;
r = 10;
mu = 3;
sigma0 = 1.5; 
sigma0_SD = 1.0e-3;
sparse_param = 1.0e-5;



[Result1, ~] = RunRep(1000, 50, 10, 0.50, 1, nrun, sigma0, 1e-3, sparse_param, 20, 5000);
[Result2, ~] = RunRep(1000, 50, 10, 0.75, 1, nrun, sigma0, 1e-3, sparse_param, 20, 5000);
[Result3, ~] = RunRep(1000, 50, 10, 1.00, 1, nrun, sigma0, 1e-3, sparse_param, 20, 5000);
[Result4, ~] = RunRep(1000, 50, 10, 1.25, 1, nrun, sigma0, 1e-3, sparse_param, 20, 5000);
[Result5, ~] = RunRep(1000, 50, 10, 1.50, 1, nrun, sigma0, 1e-3, sparse_param, 20, 5000);

[Result6, ~] = RunRep(2000, 50, 4, 3, 1, nrun, sigma0, 1e-3, sparse_param, 20, 5000);
[Result7, ~] = RunRep(2000, 50, 6, 3, 1, nrun, sigma0, 1e-3, sparse_param, 20, 5000);
[Result8, ~] = RunRep(2000, 50, 8, 3, 1, nrun, sigma0, 1e-3, sparse_param, 20, 5000);
[Result9, ~] = RunRep(2000, 50, 10, 3, 1, nrun, sigma0, 1e-3, sparse_param, 20, 5000);
[Result10, ~] = RunRep(2000, 50, 12, 3, 1, nrun, sigma0, 1e-3, sparse_param, 20, 5000);



path = strcat(pwd,'/RiAL_modified/results/SPCA/');
filename = 'SPCA_May06_data_v1.mat';
save(strcat(path,filename));


function [Result, out_ManALM] = RunRep(n, m, r, mu, start, reps, sigma0, sigma0_SD, sparse_param, inner_min_itr, inner_mxitr);
    out_ManALM = 1;
    
    T0 = []; T1 = []; T2 = []; T3 = []; T4 = [];
    S0 = []; S1 = []; S2 = []; S3 = []; S4 = [];
    F0 = []; F1 = []; F2 = []; F3 = []; F4 = [];
    I0 = []; I1 = []; I2 = []; I3 = []; I4 = [];
    V0 = []; V1 = []; V2 = []; V3 = []; V4 = [];
    TI0 = []; TI1 = []; TI2 = []; TI3 = []; TI4 = [];

    for num = start:reps

        fprintf(1, '\n                       --------- #%d/%d, n = %d, r = %d, mu = %.2f---------\n', num, reps, n, r, mu);

        A = randn(m, n);
        A = A - repmat(mean(A, 1), m, 1);
        A = A ./ repmat(sqrt(sum(A .* A)), m, 1);
        
        
        [U, S, V] = svd(A, 'econ');
        D = sort(abs(randn([m 1])) .^ 4) + 1.0e-5;

        A = U * diag(D) * V';
        A = A - repmat(mean(A, 1), m, 1);
        A = A ./ repmat(sqrt(sum(A .* A)), m, 1);
        
        [V, D] = eig(A'*A);


        eigVals = diag(D);
        [~, idx] = sort(eigVals, 'descend');
        Xpca = V(:, idx(1:r));
        Var0 = sum(Xpca.*(A'*(A*Xpca)),"all");

        [phi_init,~] = svd(randn(n,r),0);  % random intialization

        Xinitial.main = phi_init;

        tol = 1e-8; 


        % ================== ManALM ===========================
        opts_ManALM.tol = tol;
        opts_ManALM.mxitr = 100;
        opts_ManALM.lasso_constant = mu;
        opts_ManALM.subproblem_option = 1;
        opts_ManALM.sigma = sigma0;
        opts_ManALM.sparse_param = sparse_param;
        opts_ManALM.inner_mxitr = inner_mxitr;
        t_ManALM = 0;
        t = tic;
        [U_ManALM, out_ManALM]= ManALM4SPCA_modified(phi_init, opts_ManALM, A);
        t_ManALM =t_ManALM +toc(t);

        fprintf('                    ManALM & itr=%4d & f=%.8f & time=%.2fs & sparse=%.4f & l1val=%.4f \\\\ \n',out_ManALM.itr, out_ManALM.F_best,...
            t_ManALM, out_ManALM.sparse, out_ManALM.l1val);
        T4 = [T4; t_ManALM];
        F4 = [F4; out_ManALM.F_best];
        I4 = [I4; out_ManALM.itr];
        S4 = [S4; out_ManALM.sparse * 100];
        V4 = [V4; sum(U_ManALM.*(A'*(A*U_ManALM)),"all")/Var0];
        TI4 = [TI4; sum(out_ManALM.inneritr)];

        % ================== RiALSD ===========================
        opts_RiALSD.tol = tol;
        opts_RiALSD.mxitr = 5000;
        opts_RiALSD.lasso_constant = mu;
        opts_RiALSD.sigma0 = 1e-4; % sigma0_SD;
        opts_RiALSD.rho = 1.05;
        opts_RiALSD.tau1 = 0.999;
        opts_RiALSD.tau2 = 0.9;
        opts_RiALSD.subproblem_c1 = 1e-6;
        opts_RiALSD.subproblem_c2 = 1e-4;
        opts_RiALSD.inner_min_itr = 20;
        opts_RiALSD.sparse_param = sparse_param;
        opts_RiALSD.verbose = 0;

        t_RiALSD = 0;
        t = tic;
        [U_RiALSD, out_RiALSD]= RiAL_SD4SPCA(phi_init, opts_RiALSD, A);
        t_RiALSD =t_RiALSD +toc(t);

        fprintf('                    RiALSD & itr=%4d & f=%.8f & time=%.2fs & sparse=%.4f & l1val=%.4f \\\\ \n',out_RiALSD.itr, out_RiALSD.F_best,...
            t_RiALSD, out_RiALSD.sparse, out_RiALSD.l1val);
        T3 = [T3; t_RiALSD];
        F3 = [F3; out_RiALSD.F_best];
        I3 = [I3; out_RiALSD.itr];
        S3 = [S3; out_RiALSD.sparse * 100];
        V3 = [V3; sum(U_RiALSD.*(A'*(A*U_RiALSD)),"all")/Var0];
        TI3 = [TI3; sum(out_RiALSD.inneritr)];


        % =================== LSq-I ===================
        option_almssn.mu = mu;
        option_almssn.n = n; option_almssn.r = r;
        option_almssn.tau = 0.99;
        option_almssn.sigma_factor = 1.25;
        option_almssn.maxinner_iter = 300;
        option_almssn.maxnewton_iter = 10;
        option_almssn.maxcg_iter = 300;
        option_almssn.retraction = 'retr';
        option_almssn.algname = 'LSq-I';
        option_almssn.gradnorm_decay = 0.95; 
        option_almssn.gradnorm_min = 1.0e-13;
        option_almssn.verbose = 0;
        option_almssn.LS = 1;
        option_almssn.x_init = Xinitial.main;
        almssn_solver = SPCANewtonNew();
        almssn_solver = almssn_solver.init(A, option_almssn);

        tic;
        almssn_solver = almssn_solver.run(tol);
        time0 = toc;

        record0 = almssn_solver.record;
        xopt0 = almssn_solver.X; fv0 = record0.loss(end); sparsity0 = record0.sparse(end); 
        itr0 = almssn_solver.iter_num;
        fprintf('                    LS-I   & itr=%.0f & f=%.8f & time=%fs & sparse=%.4f \\\\ \n',itr0, fv0,...
            time0,  sparsity0);


        I0 = [I0; itr0];
        T0 = [T0; time0];
        F0 = [F0; fv0];
        S0 = [S0; sparsity0 * 100];

        % =================== LSq-II ===================
        option_almssn.LS = 2;
        option_almssn.x_init = Xinitial.main;
        option_almssn.algname = 'LSq-II';
        almssn_solver = SPCANewtonNew();
        almssn_solver = almssn_solver.init(A, option_almssn);

        tic;
        almssn_solver = almssn_solver.run(tol);
        time1 = toc;

        record1 = almssn_solver.record;
        xopt1 = almssn_solver.X; fv1 = record1.loss(end); sparsity1 = record1.sparse(end); 
        itr1 = almssn_solver.iter_num;
        fprintf('                    LS-II  & itr=%.0f & f=%.8f & time=%fs & sparse=%.4f \\\\ \n',itr1, fv1,...
            time1,  sparsity1);

        I1 = [I1; itr1];
        T1 = [T1; time1];
        F1 = [F1; fv1];
        S1 = [S1; sparsity1 * 100];

        % ================== RADA-RGD ====================

        Y0 = ones(size(phi_init));
        opts.n = n*r;
        opts.R = norm(mu*ones(n,r),'fro');
        opts.epsilon = tol;
        lambda_RADA = opts.epsilon/(2*opts.R);

        opts.mxitr = 50000;
        opts.lasso_constant = mu;

        opts.adbeta = 1;
        opts.nupdateU = 10;
        opts.beta0 = 1;
        opts.rho = 1.5;
        opts.verbose = 0;

        t_RADA = 0;
        t = tic;
        [U_RADA, out_RADA]= RADA_RGD4SPCA(phi_init, opts, A, lambda_RADA, Y0);
        t_RADA =t_RADA +toc(t);

        fprintf('                    RADA-RGD & itr=%4d & f=%.8f & time=%fs & sparse=%.4f \\\\ \n',out_RADA.itr, out_RADA.F_best,...
            t_RADA, out_RADA.sparse);
        T2 = [T2; t_RADA];
        F2 = [F2; out_RADA.F_best];
        I2 = [I2; out_RADA.itr];
        S2 = [S2; out_RADA.sparse * 100];
        V2 = [V2; sum(U_RADA.*(A'*(A*U_RADA)),"all")/Var0];
    end


    fprintf(1, '\n\n=========== Summary: n = %d, r = %d, mu = %.3f==========\n', n, r, mu);
    fprintf(1, 'LS-I:       time = %.3fs,  sparsity = %.2f,  loss = %.4f, iter = %.4f\n', mean(T0), mean(S0), mean(F0), mean(I0));
    fprintf(1, 'LS-II:      time = %.3fs,  sparsity = %.2f,  loss = %.4f, iter = %.4f\n', mean(T1), mean(S1), mean(F1),mean(I1));
    fprintf(1, 'RADA-RGD:   time = %.3fs,  sparsity = %.2f,  loss = %.4f, iter = %.4f\n', mean(T2), mean(S2), mean(F2),mean(I2));
    fprintf(1, 'RiALSD:     time = %.3fs,  sparsity = %.2f,  loss = %.4f, iter = %.0f, total = %.0f \n', mean(T3), mean(S3), mean(F3),mean(I3), mean(TI3));
    fprintf(1, 'ManALM:     time = %.3fs,  sparsity = %.2f,  loss = %.4f, iter = %.0f, total = %.0f \n', mean(T4), mean(S4), mean(F4),mean(I4), mean(TI4));
    % fprintf(1, '$%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ \n', mean(F2), mean(F0), mean(F1), mean(F3), mean(F4));
    % fprintf(1, '$%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ & $%.4f$ \n', mean(V2), mean(V0), mean(V1), mean(V3), mean(V4));
    % fprintf(1, '$%.0f$ & $%.0f$ & $%.0f$ & $%.0f$ & $%.0f$ \n', mean(I2), mean(I0), mean(I1), mean(I3), mean(I4));
    % fprintf(1, '$%.2f $& $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \n', mean(T2), mean(T0), mean(T1), mean(T3), mean(T4));
    % fprintf(1, '$%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ & $%.2f$ \n', mean(S2), mean(S0), mean(S1), mean(S3), mean(S4));

    Result = [mean(F2), mean(F0), mean(F1), mean(F3), mean(F4);
            mean(V2), mean(V0), mean(V1), mean(V3), mean(V4);
            mean(I2), mean(I0), mean(I1), mean(I3), mean(I4);
            mean(T2), mean(T0), mean(T1), mean(T3), mean(T4);
            mean(S2), mean(S0), mean(S1), mean(S3), mean(S4);
            mean(TI2), mean(TI0), mean(TI1), mean(TI3), mean(TI4)];

end
