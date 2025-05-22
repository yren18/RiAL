function [U_best, out]= RADA_PGD4SSC(U, opts, L, Y_now)

if isempty(U)
    error('input U is an empty matrix');
else
    [d, r] = size(U);
    U_best = zeros(d,r);
end

if ~isfield(opts, 'epsilon');      opts.epsilon = 1e-8; end
if ~isfield(opts, 'adbeta');      opts.adbeta = 0; end
% if ~isfield(opts, 'tau');       opts.tau  = 1e-3; end
% if ~isfield(opts, 'rhols');     opts.rhols  = 1e-4; end
% if ~isfield(opts, 'eta');       opts.eta  = 0.1; end
% if ~isfield(opts, 'retr');      opts.retr = 0; end
% if ~isfield(opts, 'gamma');     opts.gamma  = 0.85; end
% if ~isfield(opts, 'STPEPS');    opts.STPEPS  = 1e-10; end
% if ~isfield(opts, 'nt');        opts.nt  = 5; end
if ~isfield(opts, 'mxitr');     opts.mxitr  = 10000; end
% if ~isfield(opts, 'record');    opts.record = 0; end
% if ~isfield(opts, 'tiny');      opts.tiny = 1e-13; end
if ~isfield(opts, 'R');      opts.R = d; end
% if ~isfield(opts, 'warmstart');      opts.warmstart = 0; end
% if ~isfield(opts, 'FPCAsubgraddist'); opts.FPCAsubgraddist = 0; end
% if ~isfield(opts, 'nbreakindex'); opts.nbreakindex = 100; end

%-------------------------------------------------------------------------------
% copy parameters

% STPEPS  = opts.STPEPS;
% eta     = opts.eta;
% gamma   = opts.gamma;
% retr    = opts.retr;
% record  = opts.record;
% nt      = opts.nt;  
% crit    = ones(nt, 3);
% warmstart = opts.warmstart;
% L = varargin{1};
sparse = opts.lasso_constant;
lambda = opts.lambda;
% nbreakindex = opts.nbreakindex;
% tiny    = opts.tiny;
epsilon = opts.epsilon;

% R = opts.R;

% breakindex = 0;
% wsindex = 0;

rho = opts.rho;
% y = opts.y0;
% mu0 = opts.mu0;
% mu0 = varargin{4};
beta0 = opts.beta0;
n = opts.n;
beta0 = beta0*n*sqrt(r);
beta = beta0;

F_hist = zeros(opts.mxitr,1);
Phi_hist = zeros(opts.mxitr,1);
nls_hist = zeros(opts.mxitr,1);
beta_hist = zeros(opts.mxitr,1);
residual = 1e10;
F_histbest = inf;

% UUt = U*U';
% Y = max(-sparse,min((1/(lambda+beta)*UUt + beta/(lambda+beta)*Y_now),sparse));
% [G,~] = eigs(-(lambda+beta) * (L + Y) + UUt, r, 'largestreal');
G = U;

grad = (L+Y_now) * U;
GU = grad'*U;
dtU = grad - U*GU;     nrmG  = norm(dtU, 'fro');


%% Main Iteration

for itr = 1 : opts.mxitr
    
    npU=1;
    for nupdateU = 1:npU

    U = G;
    UUt = U*U';
    f = sum(L.*UUt,"all") + sparse * sum(sum(abs(UUt)));
    Y = max(-sparse,min((1/(lambda+beta)*UUt + beta/(lambda+beta)*Y_now),sparse));
    F = sum((L+Y).*UUt,"all") - lambda/2*norm(Y,'fro')^2 - beta/2*norm(Y-Y_now,'fro')^2;
    
    % if F <= Cval - tau*deriv || nls >= 5
    % tau;
    Phi_hist(itr) = F; 
    F_hist(itr) = f;
    if itr>=2 && abs(F_hist(itr)-F_hist(itr-1))<1e-8
        breakindex = breakindex+1;
    else
        breakindex = 0;
    end
    nls_hist(itr) = 1; %nls;
    if F_hist(itr) < F_histbest
        % breakindex = 0;
        U_best = U;
        F_histbest = F_hist(itr);
    % else
    %     breakindex = breakindex+1;

    end

    if nupdateU == npU

        if opts.adbeta 
            % T = UUt + beta*y;
            % P = sign(T) .* max(0, abs(T) - sparseparameter*(beta+lambda));
            % P = 1/(1+lambda/beta)*(P+lambda*)
            % residualnext = max(max(abs(UUt - P)));
            tau1 = 0.999;
            residualnext = max(max(abs(lambda*Y+beta*(Y_now - Y))));
            % residualnext = max(max(abs((y - ynext))));
            if (residualnext >= 1e-3 && residualnext >= tau1 * residual)% || nrmG <= lambda
                % residualnext = max(max(abs(UUt - Y)))
                % nrmG
                beta0 = 0.9*beta0;
            end
            % beta = max(mu/((itr)^rho),1e-6);
            % else
            % residualnext = max(max(abs(UUt - Y)))
            % nrmG
            beta = max(beta0/((itr)^rho),1e-9);
            residual = residualnext;
        else
            beta = beta0/((itr+1)^rho);
        end
        Y_now = Y;
        beta_hist(itr) = beta;
    end
        % beta = 1;
    % beta = 0.5*beta;
    % end
    % if itr == 2000
    %     mu = mu * 1e-9;
    % end
    % varargin{4} = beta;
    % [~, G, ~, ~] = feval(fun, U, varargin{:});

    Y = max(-sparse,min((1/(lambda+beta)*UUt + beta/(lambda+beta)*Y_now),sparse));
    % ytemp = max(-sparseparameter,min((1/(lambda+beta)*UUt + beta/(lambda+beta)*ytemp),sparseparameter));
  % ytemp = y; 
    [G,~] = eigs(-(lambda+beta) * (L + Y) + UUt, r, 'largestreal');

    % grad = 2*(L+Y_now) * U;
    % GU = grad'*U;
    % dtU = grad - U*GU;     nrmG  = norm(dtU, 'fro');

    G_temp = L+Y_now;
    % UUt_temp = U*U';
    GUUt_temp = G_temp*U*U';
    dtU_temp = GUUt_temp + GUUt_temp'-2*U*(U'*GUUt_temp);
    nrmG = norm(dtU_temp,'fro');
    % if rem(itr, 100) == 0 
    %     nrmG;
    % end
    end



    if nrmG <= epsilon && norm(Y - max(-sparse,min((UUt + Y),sparse)),'fro') <= epsilon
            break;
    end

    % GU = G'*U;
    % dtU = G - U*GU;     nrmG  = norm(dtU, 'fro');
    % S = U - UP;         XDiff = norm(S,'fro')/sqrt(d);
    % %tau = opts.tau;
    % FDiff = abs(FP-F)/(abs(FP)+1);
    % 
    % %Y = G - GP;     SY = abs(iprod(S,Y));
    % Y = dtU - dtUP;     SY = abs(iprod(S,Y));
    % if mod(itr,2)==0; tau = (norm(S,'fro')^2)/SY;
    % else tau  = SY/(norm(Y,'fro')^2); end
    % 
    % tau = max(min(tau, 1e20), 1e-20);
    % if (record >= 1)
    %     fprintf('%4d  %3.2e  %4.3e  %3.2e  %3.2e  %3.2e  %2d\n', ...
    %         itr, tau, F, nrmG, XDiff, FDiff, nls);
    % end
    % 
    % crit(itr,:) = [nrmG, XDiff, FDiff];
    % mcrit = mean(crit(itr-min(nt,itr)+1:itr, :),1);
    % 
    % if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol || all(mcrit(2:3) < 10*[xtol, ftol])
    %     out.msg = 'converge';
    %     break;
    % end
    % if beta == 0
    %     % Nonmonotone
    %     Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + F)/Q;
    %     % Monotone
    % else
    %     Cval = F + 2 * R^2 * beta;
    % end

end

out.nrmG = nrmG;
out.fval = F;
out.itr = itr;
out.obj = F_hist(1:itr);
out.Phi_hist = Phi_hist(1:itr);
out.beta_hist = beta_hist(1:itr);
out.F_best = F_histbest;

end