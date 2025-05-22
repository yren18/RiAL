function [U_best, out]= DSGM4SSCGr(U, opts, L, Y)

if isempty(U)
    error('input U is an empty matrix');
else
    [d, r] = size(U);
    U_best = zeros(d,r);
end


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
if ~isfield(opts, 'R');      opts.R = sqrt(2); end
% if ~isfield(opts, 'warmstart');      opts.warmstart = 0; end
% if ~isfield(opts, 'FPCAsubgraddist'); opts.FPCAsubgraddist = 0; end
if ~isfield(opts, 'nbreakindex'); opts.nbreakindex = 100; end
% if ~isfield(opts, 'nupdateU'); opts.nupdateU = 1; end
% if ~isfield(opts, 'monotonelinesearch'); opts.FPCAsubgraddist = 0; end

%-------------------------------------------------------------------------------
% copy parameters
c1   = opts.c1;
% STPEPS  = opts.STPEPS;
eta     = opts.eta;
% gamma   = opts.gamma;
% retr    = opts.retr;
% record  = opts.record;
% nt      = opts.nt;  
% crit    = ones(nt, 3);
% warmstart = opts.warmstart;
nbreakindex = opts.nbreakindex;
sparse = opts.lasso_constant;

% npU = opts.nupdateU;
% tiny    = opts.tiny;

R = opts.R;

% rho = opts.rho;
% % y = opts.y0;
% % mu0 = opts.mu0;
% mu0 = varargin{4};
lambda0 = opts.lambda0;
lambda = lambda0;
% n = opts.n;

F_hist = zeros(opts.mxitr,1);
nls_hist = zeros(opts.mxitr,1);
F_histbest = inf;

beta = 0;

UUt = U*U';
Ytemp = max(-sparse,min((1/(lambda+beta)*UUt + beta/(lambda+beta)*Y),sparse));
F = sum((L+Ytemp).*UUt,"all") - lambda/2*norm(Ytemp,'fro')^2 - beta/2*norm(Ytemp-Y,'fro')^2;

G =  L+Ytemp;
% XtU = X'*U;
% f = sum(XtU.*XtU,2);
% ytemp = simplexproj(y - 1/(lambda+beta)*f - lambda/(lambda+beta)*y);
% F = -f'*ytemp - lambda/2*(ytemp'*ytemp) - beta/2*((ytemp-y)'*(ytemp-y));
%F = lambda/2*(y'*y);
% yXU = ytemp.*XtU;
% G =  X * (-2 *yXU);
Ylast = Y;
Ynow = Ytemp;
% [F, G, ~, y] = feval(fun, U , varargin{:});
GUUt = (G*U)*U';

dtU = GUUt + GUUt' - 2*U*(U'*GUUt);     nrmG  = norm(dtU, 'fro');
  
Q = 1; Cval = F;  zeta = opts.zeta;


%% Main Iteration

for itr = 1 : opts.mxitr

    lambda = lambda0/(itr^(1/3));

    % for nupdateU = 1:npU

    QP = UUt;     %FP = F;   GP = G;   
    dtUP = dtU;

    nls = 1; deriv = c1*nrmG^2; %deriv


%% Line Search
    while 1

        %[U, RR] = myQR(QP - tau*dtU, r);
        [V,~] = eigs(QP - zeta*dtU, r, 'largestreal');
        U = V;
       
        UUt = U*U';
        Ytemp = max(-sparse,min((1/(lambda+beta)*UUt + beta/(lambda+beta)*Ylast),sparse));
        F = sum((L+Ytemp).*UUt,"all") - lambda/2*norm(Ytemp,'fro')^2 - beta/2*norm(Ytemp-Ylast,'fro')^2;


        if F <= Cval - zeta*deriv || nls >= 5

            % tau;

            fval = sum(L.*UUt,"all") + sparse * sum(sum(abs(UUt)));

            F_hist(itr) = fval;
            nls_hist(itr) = nls;

            if itr>=2 && F_histbest-F_hist(itr) < 1e-8
                 breakindex = breakindex+1;
             else
                 breakindex = 0;
             end

            if F_hist(itr) < F_histbest - 1e-8 
                    
                    % norm(U'*U-eye(r),'fro')
                    if norm(U'*U-eye(r),'fro')<1e-8
                    % breakindex = 0;
                    U_best = U;
                    F_histbest = F_hist(itr);
                    end

            end

            break;
        end
        zeta = eta*zeta;          nls = nls+1;
    end 

    %New Gradient
    % yXU = ytemp.*XtU;%FPCA
    % G =  X * (-2 *yXU);

    G =  L+Ytemp;

    % if nupdateU == npU
    
    Ylast = Ynow;
    % ynow = simplexproj(ylast - 1/(lambda+beta)*f - lambda/(lambda+beta)*ylast);
    Ynow = max(-sparse,min((1/(lambda+beta)*UUt + beta/(lambda+beta)*Ylast),sparse));

    % F = -f'*ynow - lambda/2*(ynow'*ynow) - beta/2*((ynow-ylast)'*(ynow-ylast));
    F = sum((L+Ynow).*UUt,"all") - lambda/2*norm(Ynow,'fro')^2 - beta/2*norm(Ynow-Ylast,'fro')^2;


    % yXU = ynow.*XtU;
    % G =  X * (-2 *yXU);
    %G =  2*(L+Ynow)*U;

    GUUt = (G*U)*U';

    dtU = GUUt + GUUt' - 2*U*(U'*GUUt);

    nrmG  = norm(dtU, 'fro');
    S = UUt - QP;         
    % XDiff = norm(S,'fro')/sqrt(d);
    % %tau = opts.tau;     
    % FDiff = abs(FP-F)/(abs(FP)+1);
    
    %Y = G - GP;     SY = abs(iprod(S,Y));
    Y = dtU - dtUP;     SY = abs(iprod(S,Y));
    if mod(itr,2)==0
        zeta = (norm(S,'fro')^2)/SY;
    else 
        zeta  = SY/(norm(Y,'fro')^2); 
    end

    zeta = max(min(zeta, 1e20), 1e-20);
    
    % if (record >= 1)
    %     fprintf('%4d  %3.2e  %4.3e  %3.2e  %3.2e  %3.2e  %2d\n', ...
    %         itr, tau, F, nrmG, XDiff, FDiff, nls);
    % end
    % 
    % crit(itr,:) = [nrmG, XDiff, FDiff];
    % mcrit = mean(crit(itr-min(nt,itr)+1:itr, :),1);

    % if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol || all(mcrit(2:3) < 10*[xtol, ftol])
    %     out.msg = 'converge';
    %     break;
    % end

    % if beta == 0
    %     % Nonmonotone
    %     Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + F)/Q;
    %     % Monotone
    % else
        Cval = F + 2 * R^2 * beta;
    % end

    % end

    if breakindex >= nbreakindex
         break;
    end

    
    % U_best = U_best;
end

out.nrmG = nrmG;
out.fval = F;
out.itr = itr;
out.obj = F_hist(1:itr);
out.nls_hist = nls_hist(1:itr);
out.F_best = F_histbest;


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