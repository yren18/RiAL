function [U, out]= ManALM4CM_modified(U, opts, H)

[d, r] = size(U);

if ~isfield(opts, 'tol');      opts.tol = 1e-8; end
% parameters for control the linear approximation in line search,
if ~isfield(opts, 'sigma');        opts.sigma = 1; end
if ~isfield(opts, 'mxitr');     opts.mxitr  = 1000; end
if ~isfield(opts, 'lasso_constant'); opts.lasso_constant = 1; end
if ~isfield(opts, 'sparse_param'); opts.sparse_param = 1e-5; end
if ~isfield(opts, 'inner_mxitr'); opts.inner_mxitr = 5000; end
if ~isfield(opts, 'verbose'); opts.verbose = 0; end

tol = opts.tol;
sigma0 = opts.sigma;
mxitr = opts.mxitr;
sparse = opts.lasso_constant;
sparse_param = opts.sparse_param;
inner_mxitr = opts.inner_mxitr;
verbose = opts.verbose;

Y = zeros(d,r);
F_hist = zeros(mxitr,1);
inneritr_hist = zeros(mxitr,1);
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
    sigma = 1*sigma0^itr;
    inner_opts.mxitr = inner_mxitr;
    inner_opts.gtol = 2*sparse/sigma;

    inner_opts.record = 0;
    inner_opts.xtol = 1.0e-20;
    inner_opts.ftol = 1.0e-20;

    % Solve the subproblem using GBB
    
    [U, info] = OptStiefelGBB(U, @fun, inner_opts, H, Y, sigma, sparse);
    HU = H*U;
    curF = sum(U.*HU,"all") + sparse*sum(abs(U),'all');
    F_hist(itr) = curF;
    inneritr_hist(itr) = info.itr;

    % Update dual variable
    P = l1_prox(U + Y / sigma, sparse / sigma); % auxiliary primal variable
    Y = Y + sigma * (U - P);                   % dual variable 

    % prs = max(max(abs(U-P)));
    prs = norm(U-P,'fro')/(1+norm(U,'fro')+norm(Y,'fro'));
    
    G = 2 * HU + Y;%
    GU = G'*U;
    dtU = G - U*GU;     nrmG  = norm(dtU, 'fro');
    etad = nrmG/(1+norm(G-Y,'fro'));

    etaC = norm(Y-max(-sparse,min(Y+U,sparse)),'fro');
    % etaC = etaC/(1+norm(Yh,'fro'));;
    cur_time = toc(starting_time);
    if verbose
        fprintf("   itr: %d:; fval: %.5f, time: %.2f, etad: %.5f, etaC: %.5f, prs: %.5f, sigma: %.5f, inner_itr: %d\n", itr, curF, cur_time, etad, etaC, prs, sigma, info.itr);
    end

    if max([etad, prs, etaC]) <= tol
        break
    end

    
end
[etad, prs, etaC];
out.nrmG = nrmG;
out.F_best = sum(U.*HU,"all") + sparse*sum(abs(U),'all');
out.l1val = sparse*sum(abs(U),'all');
out.itr = itr;
out.sparse = sum(sum(abs(U) < sparse_param)) / (d * r);
out.F_hist = F_hist(1:itr,:);
out.inneritr = inneritr_hist(1:itr,:);

end

function [F, G] = fun(U, H, Y, sigma, sparse)

HU = H*U;
PY = sigma * l1_moreau(U + Y / sigma, sparse / sigma);
F = sum(U.*HU,"all") + sum(PY,'all');

T = U + Y / sigma;
G = 2 * HU + sigma * (T - l1_prox(T, sparse / sigma));

end

function ret = l1_moreau(X, lam)
p = (abs(X) < lam);
ret = p .* (0.5 * X .* X) + (1 - p) .* (lam * abs(X) - 0.5 * lam * lam);
end
function ret = l1_prox(X, lam)
ret = sign(X) .* max(0, abs(X) - lam);
end