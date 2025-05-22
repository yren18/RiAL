function [U, out]= ManALM4SPCA(U, opts, A)

[d, r] = size(U);

 if ~isfield(opts, 'tol');      opts.tol = 1e-8; end
% parameters for control the linear approximation in line search,
 if ~isfield(opts, 'beta0');       opts.beta0  = 1; end
% if ~isfield(opts, 'rhols');     opts.rhols  = 1e-4; end
% if ~isfield(opts, 'eta');       opts.eta  = 0.1; end
% if ~isfield(opts, 'retr');      opts.retr = 0; end
% if ~isfield(opts, 'gamma');     opts.gamma  = 0.85; end
% if ~isfield(opts, 'STPEPS');    opts.STPEPS  = 1e-10; end
if ~isfield(opts, 'sigma');        opts.sigma = 1; end
if ~isfield(opts, 'mxitr');     opts.mxitr  = 1000; end
if ~isfield(opts, 'new');     opts.new  = 0; end
if ~isfield(opts, 'subproblem_option');      opts.subproblem_option = 2; end
% if ~isfield(opts, 'warmstart');      opts.warmstart = 0; end
% if ~isfield(opts, 'FPCAsubgraddist'); opts.FPCAsubgraddist = 0; end
% if ~isfield(opts, 'nbreakindex'); opts.nbreakindex = 100; end
if ~isfield(opts, 'lasso_constant'); opts.lasso_constant = 1; end
% if ~isfield(opts, 'nupdateU'); opts.nupdateU = 1; end

beta0 = opts.beta0;
tol = opts.tol;
sigma0 = opts.sigma;
mxitr = opts.mxitr;
sparse = opts.lasso_constant;
subproblem_option = opts.subproblem_option;
new = opts.new;

Y = zeros(d,r);
F_hist = zeros(mxitr,1);
inneritr_hist = zeros(mxitr,1);
beta_hist = zeros(mxitr,1); 

%%%
% beta0 = sparse/(pi^2/6*norm(U,"fro"));

for itr = 1:mxitr

    % GBB Setup
    if subproblem_option == 1
        sigma = 1*sigma0^itr;
        inner_opts.mxitr = 5000;
        inner_opts.gtol = 2*sparse/sigma;
    elseif subproblem_option == 2
        sigma = sigma0*(1.3^(itr/3));
        inner_opts.mxitr = min(ceil(1.3^itr),2000);
        inner_opts.gtol = 1e-20;
    end
    inner_opts.record = 0;
    inner_opts.xtol = 1.0e-20;
    inner_opts.ftol = 1.0e-20;

    % Solve the subproblem using GBB
    
    [U, info] = OptStiefelGBB(U, @fun, inner_opts, A, Y, sigma, sparse);
    % [U0,~] = svd(randn(d,r),0);
    % inner_opts.record = 1;
    % [U1, info1] = OptStiefelGBB(U0, @fun, inner_opts, A, Y, sigma, sparse);

    AU = A*U;
    F_hist(itr) = -sum(AU.*AU,"all") + sparse*sum(abs(U),'all');
    inneritr_hist(itr) = info.itr;

    % Update dual variable
    
    P = l1_prox(U + Y / sigma, sparse / sigma);

    beta = beta0 * min(1, log(2)^2 / (norm(U-P,'fro') * (itr+1)^2 * log(itr+2)));
    beta_hist(itr) = beta;
    
    Yh = Y + sigma * (U - P);
    Y = Y + beta * (U - P);

    if new == 1
        Y = Yh;
    end
    %Y = 0*Y; 
    % normYh = norm(Yh,'fro')
    % normY = norm(Y,'fro')
    %Y = Yh;
    % 
    % prs = max(max(abs(U-P)));
    prs = norm(U-P,'fro')/(1+norm(U,'fro')+norm(Y,'fro'));
    
    % AU = A*U;
    % Yh = Y;
    AtAU = A'*AU;
    G = -2 * AtAU + Yh;%
    GU = G'*U;
    dtU = G - U*GU;     nrmG  = norm(dtU, 'fro');
    etad = nrmG/(1+norm(G-Yh,'fro'));

    etaC = norm(Yh-max(-sparse,min(Yh+U,sparse)),'fro');
    % etaC = etaC/(1+norm(Yh,'fro'));;

    if max([etad, prs, etaC]) <= tol
        break
    end

    
end
[etad, prs, etaC];
out.nrmG = nrmG;
out.F_best = -sum(AU.*AU,"all") + sparse*sum(abs(U),'all');
out.l1val = sparse*sum(abs(U),'all');
out.itr = itr;
out.sparse = sum(sum(abs(U) < 1.0e-5)) / (d * r);
out.F_hist = F_hist(1:itr,:);
out.inneritr = inneritr_hist(1:itr,:);
out.beta_hist = beta_hist(1:itr,:);

end

function [F, G] = fun(U, A, Y, sigma, sparse)

AU = A*U;
PY = sigma * l1_moreau(U + Y / sigma, sparse / sigma);
F = -sum(AU.*AU,"all") + sum(PY,'all');

T = U + Y / sigma;
G = -2 * A'*AU + sigma * (T - l1_prox(T, sparse / sigma));

end

function ret = l1_moreau(X, lam)
p = (abs(X) < lam);
ret = p .* (0.5 * X .* X) + (1 - p) .* (lam * abs(X) - 0.5 * lam * lam);
end
function ret = l1_prox(X, lam)
ret = sign(X) .* max(0, abs(X) - lam);
end