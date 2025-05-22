function [U, out]= RiAL_SD4SSCGr(U, opts, L)
    %% load input
    if ~isfield(opts, 'tol');      opts.tol = 1e-8; end
    if ~isfield(opts, 'sigma');        opts.sigma = 1; end
    if ~isfield(opts, 'mxitr');     opts.mxitr  = 5000; end
    if ~isfield(opts, 'lasso_constant'); opts.lasso_constant = 1; end
    if ~isfield(opts, 'rho'); opts.rho = 1.05; end
    if ~isfield(opts, 'tau1'); opts.tau1 = 0.999; end
    if ~isfield(opts, 'tau2'); opts.tau2 = 0.9; end
    if ~isfield(opts, 'subproblem_c1'); opts.subproblem_c1 = 1e-2; end
    if ~isfield(opts, 'subproblem_c2'); opts.subproblem_c2 = 1e-4; end
    if ~isfield(opts, 'sparse_param'); opts.sparse_param = 1e-5; end
    if ~isfield(opts, 'inner_min_itr'); opts.inner_min_itr = 20; end
    if ~isfield(opts, 'verbose'); opts.verbose = 0; end
    % copy parameters
    [d, r] = size(U);
    tol = opts.tol;
    sigma0 = opts.sigma0;
    mxitr = opts.mxitr;
    sparse = opts.lasso_constant;
    rho = opts.rho;
    tau1 = opts.tau1;
    tau2 = opts.tau2;
    subproblem_c1 = opts.subproblem_c1;
    subproblem_c2 = opts.subproblem_c2;
    sparse_param = opts.sparse_param;
    inner_min_itr = opts.inner_min_itr;
    verbose = opts.verbose;


    sigma = sigma0;
    %% initialization
    Y = ones(d,d);
    F_hist = zeros(mxitr,1);
    inneritr_hist = zeros(mxitr,1);
    prs = 1e9;
    starting_time = tic();
    for itr = 1:mxitr

        % GBB Setup
        % if subproblem_option == 1
        %     sigma = 1*sigma0^itr;
        %     inner_opts.mxitr = 5000;
        %     inner_opts.gtol = 2*sparse/sigma;
        % elseif subproblem_option == 2
        %     sigma = sigma0*(1.3^(itr/3));
        %     inner_opts.mxitr = min(ceil(1.3^itr),2000);
        %     inner_opts.gtol = 1e-20;
        % end
        %% Solve the subproblem using GBB

        inner_opts.mxitr = 5000;
        inner_opts.gtol = 1e-6;
        inner_opts.record = 0;
        inner_opts.xtol = 1.0e-20;
        inner_opts.ftol = 1.0e-20;
        inner_opts.stop_criterion = 2;
        inner_opts.subproblem_c1 = subproblem_c1;
        inner_opts.subproblem_c2 = subproblem_c2;
        inner_opts.inner_min_itr = inner_min_itr;

        [U, info] = OptStiefelGBB_modified(U, @fun, inner_opts, L, Y, sigma, sparse);
        UUt = U*U';
        curF = sum(L.*UUt,"all") + sparse*sum(abs(UUt),'all');
        F_hist(itr) = curF;
        inneritr_hist(itr) = info.itr;

        %% Update dual variable
        P = l1_prox(UUt + Y / sigma, sparse / sigma); % auxiliary primal variable
        Y = Y + sigma * (UUt - P);                   % dual variable 

        %% update penalty parameter sigma
        % cur_primal_residual = max(max(abs(U-P))); % norm(U - P, 'fro');
        % if cur_primal_residual >= 1e-3 && cur_primal_residual >= tau1 * primal_residual
        %     sigma0 = sigma*(1/tau2);
        %     fprintf("jump");
        % end
        % sigma = sigma0*(itr^rho);
        % fprintf("          sigma0: %.8f, rest: %.3f\n", sigma0, itr^rho);
        % primal_residual = cur_primal_residual;
        % prs = max(max(abs(U-P)));
        cur_prs = norm(UUt-P,'fro'); 
        % cur_prs = norm(U-P,'fro')/(1+norm(U,'fro')+norm(Y,'fro'));
        jump_threshold = 1e-3;
        if prs >= jump_threshold && cur_prs >= tau1 * prs
            sigma0 = sigma0*(1/tau2);
        end
        sigma = sigma0*(itr^rho);
        prs = cur_prs;
        G = L+Y;
        % G = (G*U)*U';
        GUUt = G*UUt;
        % dtU = GUUt + GUUt' - 2*U*(U'*GUUt);     nrmG  = norm(dtU, 'fro');
        dtU = GUUt + GUUt' - 2*UUt*GUUt;     nrmG  = norm(dtU, 'fro');
        etad = nrmG/(1+norm(G-Y,'fro'));

        etaC = norm(Y-max(-sparse,min(Y+UUt,sparse)),'fro');
        etaC = etaC/(1+norm(Y,'fro'));
        etaP = norm(UUt-P,'fro')/(1+norm(UUt,'fro')+norm(P,'fro'));
        cur_time = toc(starting_time);
        if verbose
            fprintf("   itr: %d:; fval: %.5f, time: %.2f, etad: %.8f, etaC: %.8f, prs: %.8f, sigma: %.5f, sigma0: %.5f, inner_itr: %d\n", itr, curF, cur_time, etad, etaC, prs, sigma, sigma0, info.itr);
        end
        if max([etad, etaP, etaC]) <= tol
            break
        end
        % if max([etad, prs, etaC]) <= tol
        %     break
        % end
    end
    [etad, prs, etaC];
    out.nrmG = nrmG;
    out.F_best = sum(L.*UUt,"all") + sparse*sum(abs(UUt),'all');
    out.l1val = sparse*sum(abs(U),'all');
    out.itr = itr;
    out.sparse = sum(sum(abs(U) < sparse_param)) / (d * r);
    out.F_hist = F_hist(1:itr,:);
    out.inneritr = inneritr_hist(1:itr,:);
end
%% utils
function [F, G] = fun(U, L, Y, sigma, sparse)
    UUt = U*U';
    PY = sigma * l1_moreau(UUt + Y / sigma, sparse / sigma);
    F = sum(L.*UUt,"all") + sum(PY,'all');

    T = UUt + Y / sigma;
    TG = T - l1_prox(T, sparse / sigma);
    % G = (L+L')*U + sigma * (T - l1_prox(T, sparse / sigma));
    G = (L+L')*U + sigma*(TG+TG')*U;
end
function ret = l1_moreau(X, lam)
    p = (abs(X) < lam);
    ret = p .* (0.5 * X .* X) + (1 - p) .* (lam * abs(X) - 0.5 * lam * lam);
end
function ret = l1_prox(X, lam)
    ret = sign(X) .* max(0, abs(X) - lam);
end



% 
% function [U, out]= RiAL_SD4SSCGr(U, opts, L)
%     %% load input
%     if ~isfield(opts, 'tol');      opts.tol = 1e-8; end
%     if ~isfield(opts, 'sigma');        opts.sigma = 1; end
%     if ~isfield(opts, 'mxitr');     opts.mxitr  = 5000; end
%     if ~isfield(opts, 'lasso_constant'); opts.lasso_constant = 1; end
%     if ~isfield(opts, 'rho'); opts.rho = 1.05; end
%     if ~isfield(opts, 'tau1'); opts.tau1 = 0.999; end
%     if ~isfield(opts, 'tau2'); opts.tau2 = 0.9; end
%     if ~isfield(opts, 'subproblem_c1'); opts.subproblem_c1 = 1e-2; end
%     if ~isfield(opts, 'subproblem_c2'); opts.subproblem_c2 = 1e-4; end
%     if ~isfield(opts, 'sparse_param'); opts.sparse_param = 1e-5; end
%     if ~isfield(opts, 'inner_min_itr'); opts.inner_min_itr = 20; end
%     if ~isfield(opts, 'verbose'); opts.verbose = 0; end
%     % copy parameters
%     [d, r] = size(U);
%     tol = opts.tol;
%     sigma0 = opts.sigma0;
%     mxitr = opts.mxitr;
%     sparse = opts.lasso_constant;
%     rho = opts.rho;
%     tau1 = opts.tau1;
%     tau2 = opts.tau2;
%     subproblem_c1 = opts.subproblem_c1;
%     subproblem_c2 = opts.subproblem_c2;
%     sparse_param = opts.sparse_param;
%     inner_min_itr = opts.inner_min_itr;
%     verbose = opts.verbose;
% 
% 
%     sigma = sigma0;
%     %% initialization
%     Y = zeros(d,d);
%     F_hist = zeros(mxitr,1);
%     inneritr_hist = zeros(mxitr,1);
%     prs = 1e9;
%     starting_time = tic();
%     for itr = 1:mxitr
% 
%         % GBB Setup
%         % if subproblem_option == 1
%         %     sigma = 1*sigma0^itr;
%         %     inner_opts.mxitr = 5000;
%         %     inner_opts.gtol = 2*sparse/sigma;
%         % elseif subproblem_option == 2
%         %     sigma = sigma0*(1.3^(itr/3));
%         %     inner_opts.mxitr = min(ceil(1.3^itr),2000);
%         %     inner_opts.gtol = 1e-20;
%         % end
%         %% Solve the subproblem using GBB
% 
%         inner_opts.mxitr = 5000;
%         inner_opts.gtol = 1e-6;
%         inner_opts.record = 0;
%         inner_opts.xtol = 1.0e-20;
%         inner_opts.ftol = 1.0e-20;
%         inner_opts.stop_criterion = 2;
%         inner_opts.subproblem_c1 = subproblem_c1;
%         inner_opts.subproblem_c2 = subproblem_c2;
%         inner_opts.inner_min_itr = inner_min_itr;
% 
%         [U, info] = OptStiefelGBB_modified(U, @fun, inner_opts, L, Y, sigma, sparse);
%         UUt = U*U';
%         curF = sum(L.*UUt,"all") + sparse*sum(abs(UUt),'all');
%         F_hist(itr) = curF;
%         inneritr_hist(itr) = info.itr;
% 
%         %% Update dual variable
%         P = l1_prox(UUt + Y / sigma, sparse / sigma); % auxiliary primal variable
%         Y = Y + sigma * (UUt - P);                   % dual variable 
% 
%         %% update penalty parameter sigma
%         % cur_primal_residual = max(max(abs(U-P))); % norm(U - P, 'fro');
%         % if cur_primal_residual >= 1e-3 && cur_primal_residual >= tau1 * primal_residual
%         %     sigma0 = sigma*(1/tau2);
%         %     fprintf("jump");
%         % end
%         % sigma = sigma0*(itr^rho);
%         % fprintf("          sigma0: %.8f, rest: %.3f\n", sigma0, itr^rho);
%         % primal_residual = cur_primal_residual;
%         % prs = max(max(abs(U-P)));
%         cur_prs = norm(UUt-P,'fro'); 
%         % cur_prs = norm(U-P,'fro')/(1+norm(U,'fro')+norm(Y,'fro'));
%         jump_threshold = 1e-3;
%         if prs >= jump_threshold && cur_prs >= tau1 * prs
%             sigma0 = sigma0*(1/tau2);
%         end
%         sigma = sigma0*(itr^rho);
% 
%         P_star = l1_prox(UUt + Y / sigma, sparse / sigma); % auxiliary primal variable
%         Y_star = sigma*UUt + P - sigma*P_star;
%         prs = cur_prs;
%         G = L+Y_star;
%         % G = (G*U)*U';
%         GUUt = G*UUt;
%         % dtU = GUUt + GUUt' - 2*U*(U'*GUUt);     nrmG  = norm(dtU, 'fro');
%         dtU = GUUt + GUUt' - 2*UUt*GUUt;     nrmG  = norm(dtU, 'fro');
%         etad = nrmG/(1+norm(G-Y_star,'fro'));
% 
%         etaC = norm(Y_star-max(-sparse,min(Y_star+UUt,sparse)),'fro');
%         etaC = etaC/(1+norm(Y_star,'fro'));
%         etaP = norm(UUt-P_star,'fro')/(1+norm(UUt,'fro')+norm(P_star,'fro'));
%         cur_time = toc(starting_time);
%         if verbose
%             fprintf("   itr: %d:; fval: %.5f, time: %.2f, etad: %.8f, etaC: %.8f, prs: %.8f, sigma: %.5f, sigma0: %.5f, inner_itr: %d\n", itr, curF, cur_time, etad, etaC, prs, sigma, sigma0, info.itr);
%         end
%         if max([etad, etaP, etaC]) <= tol
%             break
%         end
%         % if max([etad, prs, etaC]) <= tol
%         %     break
%         % end
%     end
%     [etad, prs, etaC];
%     out.nrmG = nrmG;
%     out.F_best = sum(L.*UUt,"all") + sparse*sum(abs(UUt),'all');
%     out.l1val = sparse*sum(abs(U),'all');
%     out.itr = itr;
%     out.sparse = sum(sum(abs(U) < sparse_param)) / (d * r);
%     out.F_hist = F_hist(1:itr,:);
%     out.inneritr = inneritr_hist(1:itr,:);
% end
% %% utils
% function [F, G] = fun(U, L, Y, sigma, sparse)
%     UUt = U*U';
%     PY = sigma * l1_moreau(UUt + Y / sigma, sparse / sigma);
%     F = sum(L.*UUt,"all") + sum(PY,'all');
% 
%     T = UUt + Y / sigma;
%     TG = T - l1_prox(T, sparse / sigma);
%     % G = (L+L')*U + sigma * (T - l1_prox(T, sparse / sigma));
%     G = (L+L')*U + sigma*(TG+TG')*U;
% end
% function ret = l1_moreau(X, lam)
%     p = (abs(X) < lam);
%     ret = p .* (0.5 * X .* X) + (1 - p) .* (lam * abs(X) - 0.5 * lam * lam);
% end
% function ret = l1_prox(X, lam)
%     ret = sign(X) .* max(0, abs(X) - lam);
% end