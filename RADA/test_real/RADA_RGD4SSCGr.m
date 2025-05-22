function [U_best, out]= RADA_RGD4SSCGr(U, opts, L, Y)
%sparseparameter = 5e-3;

if isempty(U)
    error('input U is an empty matrix');
else
    [d, r] = size(U);
    U_best = zeros(d,r);
end

if ~isfield(opts, 'epsilon');      opts.epsilon = 1e-8; end

% parameters for control the linear approximation in line search,
if ~isfield(opts, 'zeta');       opts.zeta  = 1e-3; end
if ~isfield(opts, 'c1');     opts.c1  = 1e-4; end
if ~isfield(opts, 'eta');       opts.eta  = 0.1; end
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
if ~isfield(opts, 'nbreakindex'); opts.nbreakindex = 100; end
if ~isfield(opts, 'lasso_constant'); opts.lasso_constant = 5e-3; end
if ~isfield(opts, 'nupdateU'); opts.nupdateU = 1; end
% if ~isfield(opts, 'monotonelinesearch'); opts.FPCAsubgraddist = 0; end

%-------------------------------------------------------------------------------
% copy parameters
% xtol    = opts.xtol;
% gtol    = opts.gtol;
% ftol    = opts.ftol;
c1 = opts.c1;
% STPEPS  = opts.STPEPS;
eta     = opts.eta;
% gamma   = opts.gamma;
% retr    = opts.retr;
% record  = opts.record;
% nt      = opts.nt;
% crit    = ones(nt, 3);
% warmstart = opts.warmstart;
% nbreakindex = opts.nbreakindex;
sparseparameter = opts.lasso_constant;
npU = opts.nupdateU;
% tiny    = opts.tiny;
epsilon = opts.epsilon;

R = opts.R;
lambda = opts.lambda;
beta0 = opts.beta0;
n = opts.n;
beta0 = beta0*n*sqrt(r);
beta = beta0;
rho = opts.rho;
% y = opts.y0;
% mu0 = opts.mu0;
% mu0 = varargin{4};


F_hist = zeros(opts.mxitr,1);
nls_hist = zeros(opts.mxitr,1);
beta_hist = zeros(opts.mxitr,1);
F_histbest = inf;

% [F, G, ~, ~] = feval(fun, U , varargin{:});
% GU = G'*U;
% 
% dtU = G - U*GU;     nrmG  = norm(dtU, 'fro');

% L = varargin{1};

residual = 1e10;

% beta = beta;
% itr_beta = 1;
Y_now = Y;
UUt = U*U';
Y = max(-sparseparameter,min((1/(lambda+beta)*UUt + beta/(lambda+beta)*Y_now),sparseparameter));
F = sum((L+Y).*UUt,"all") - lambda/2*norm(Y,'fro')^2 - beta/2*norm(Y-Y_now,'fro')^2;
G = L+Y;
GUUt = (G*U)*U';
dtU = GUUt + GUUt' - 2*U*(U'*GUUt);    nrmG  = norm(dtU, 'fro');

Q = 1; Cval = F;  zeta = opts.zeta;
%% Main Iteration

for itr = 1 : opts.mxitr

    % npU = 50;
    for nupdateU = 1:npU

        UP = UUt;     % FP = F;   GP = G;   
        dtUP = dtU;

        nls = 1; deriv = c1*nrmG^2;

        %% Line Search
        while 1

            %[U, RR] = myQR(UP - tau*dtU, r);
            [V,~] = eigs(UP - zeta*dtU, r, 'largestreal');
            U = V;

            UUt = U*U';

            Y = max(-sparseparameter,min((1/(lambda+beta)*UUt + beta/(lambda+beta)*Y_now),sparseparameter));

            F = sum((L+Y).*UUt,"all") - lambda/2*norm(Y,'fro')^2 - beta/2*norm(Y-Y_now,'fro')^2;


            if F <= Cval - zeta*deriv || nls >= 5

                f = sum(L.*UUt,"all") + sparseparameter * sum(sum(abs(UUt)));

                F_hist(itr) = f;
                nls_hist(itr) = nls;

                if itr>=2 && F_histbest-F_hist(itr) < 1e-8
                    breakindex = breakindex+1;
                else
                    breakindex = 0;
                end

                if F_hist(itr) < F_histbest - 1e-8

                    % breakindex = 0;
                    U_best = U;
                    Y_best = Y;
                    F_histbest = F_hist(itr);
                    % lambda2 = lambda;
                    % else
                    %     breakindex = breakindex+1;
                end

                break;
            end
            zeta = eta*zeta;          nls = nls+1;
        end

        % Update Y_k, beta_k
        if nupdateU == npU
            
            tau1 = 0.999;

            if opts.adbeta 
                % T = UUt + beta*Y_now;
                % P = sign(T) .* max(0, abs(T) - sparseparameter*(beta));
                % residualnext = max(max(abs(UUt - P)));
                residualnext = max(max(abs(lambda*Y+beta*(Y_now - Y))));
                if (residualnext >= 1e-3 && residualnext >= tau1 * residual) %|| nrmG <= lambda
                % residualnext = max(max(abs(UUt - Y)))
                % nrmG
                beta0 = 0.9*beta0;
                end
             
                beta = max(beta0/((itr)^rho),1e-9);
                residual = residualnext;
            else
                beta = beta0/((itr+1)^rho);
            end
            
            beta_hist(itr) = beta;
            Y_now = Y;

            Y = max(-sparseparameter,min((1/(lambda+beta)*UUt + beta/(lambda+beta)*Y_now),sparseparameter));
        
        end

        G = L+Y;
        GUUt = (G*U)*U';
        dtU = GUUt + GUUt' - 2*U*(U'*GUUt);     nrmG  = norm(dtU, 'fro');
        S = UUt - UP;         
        % XDiff = norm(S,'fro')/sqrt(d);
        % %tau = opts.tau;
        % FDiff = abs(FP-F)/(abs(FP)+1);

        %Y = G - GP;     SY = abs(iprod(S,Y));
        y = dtU - dtUP;     SY = abs(iprod(S,y));
        if mod(nupdateU,2)==0 
            zeta = (norm(S,'fro')^2)/SY;
        else 
            zeta  = SY/(norm(y,'fro')^2); 
        end

        zeta = max(min(zeta, 1e20), 1e-20);

        % if (record >= 1)
        %     fprintf('%4d  %3.2e  %4.3e  %3.2e  %3.2e  %3.2e  %2d\n', ...
        %         itr, tau, F, nrmG, XDiff, FDiff, nls);
        % end
        % 
        % crit(itr,:) = [nrmG, XDiff, FDiff];
        % mcrit = mean(crit(itr-min(nt,itr)+1:itr, :),1);


        % if beta == 0
        %     % Nonmonotone
        %     Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + F)/Q;
        %     % Monotone
        % else
            Cval = F + 2 * R^2 * beta;
        % end


    end

    % if breakindex >= nbreakindex || wsindex >= 5000
    %         break;
    % end
    % 
 
    if nrmG <= epsilon && norm(Y - max(-sparseparameter,min((UUt + Y),sparseparameter)),'fro') <= epsilon
            break;
    end

end

out.nrmG = nrmG;
out.fval = F;
out.itr = itr;
out.obj = F_hist(1:itr);
out.nls_hist = nls_hist(1:itr);
out.beta_hist = beta_hist(1:itr);
out.F_best = F_histbest;
% out.lambda = lambda2;
out.Y_best = Y_best;

out.Y_end = Y;
out.U_end = U;
% out.lambda_end = lambda;

end

function a = iprod(x,y)
%a = real(sum(sum(x.*y)));
a = real(sum(sum(conj(x).*y)));
end

% function [Q, RR] = myQR(XX,k)
% [Q, RR] = qr(XX, 0);
% diagRR = sign(diag(RR)); ndr = diagRR < 0;
% if nnz(ndr) > 0
%     Q = Q*spdiags(diagRR,0,k,k);
%     %Q(:,ndr) = Q(:,ndr)*(-1);
% end
% 
% end