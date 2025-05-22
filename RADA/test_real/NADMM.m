function  [U,sparsity_multi, out] = NADMM(U, L, sparse, opts)

n = size(U,1);
k = size(U,2);
% Inital value of P
P = U*U';
sigma0 = 0.1;
sigma = sigma0;
lambda0 = opts.lambda0;%1e-8;
mxitr = opts.mxitr;
nbreakindex = opts.nbreakindex;


% Initial value of Y
Y = ones(n,n);
Y = Y + Y';
% Y = max(-entropy_penalty,min(Y,entropy_penalty));

% fprintf('mu value is %f\n',mu)
% maximum value of mu
% objective value
% function_val = [];
F_hist = zeros(mxitr, 1);
sigma_hist = zeros(mxitr, 1);
F_best = 1e10;

% err = 10; rep = 0;
for itr =1:mxitr

    % Update U
    lambda = lambda0/itr^(1/3);
    A = P - (L - Y)/sigma;

    [V,~] = eigs(A, k, 'largestreal');
    U = V;

    UUt = U*U';

    % Update P
    % componentwisely
    Q = UUt - Y/sigma;
    top_value = sparse/sigma;
    bottom_value = max(abs(Q),sparse*lambda+sparse/sigma);

    P = (1 - top_value./bottom_value).*Q;

    % % Update Y, the dual variable
    Y = Y + sigma*(P - UUt);

    % % Compute the primal residual
    % primal_resid = norm(UUt-Pnext,'fro');
    % % Compute the dual residual
    % dual_resid = norm(Pnext - P,'fro');

    % fprintf('primal_resid = %f, dual_resid = %f\n',...
    %     primal_resid, dual_resid)

    % automatically vary the penalty parameter
    %
    % if primal_resid > 10*dual_resid
    %     mu = 2*mu;
    % elseif dual_resid > 10*primal_resid
    %     mu = mu/2;
    % else
    %     mu = mu*1;
    % end
    % mu
    % mu = 50;
    % mu = itr^1.1/(1e-5*40000^2*sqrt(3));
    % mu = 1.01*mu;
    sigma = sigma0*itr^(1/3);
    sigma_hist(itr) = sigma;

    % err = norm(U-Unext,'fro') + norm(P-Pnext,'fro') + norm(Y-Ynext,'fro');
    % norm(P-U*U','fro')
    % F_hist(itr) = objective_function(Lsum,U,beta);
    F_hist(itr) = sum(L.*UUt,"all") + sparse * sum(sum(abs(UUt)));

    if F_hist(itr) < F_best - 1e-8

        F_best = F_hist(itr);
        breakindex = 0;
        U_best = U;

    else

        breakindex = breakindex+1;
    end
    % if itr>=2 && abs(F_hist(itr)-F_hist(itr-1))<1e-8
    %     breakindex = breakindex+1;
    % else
    %     breakindex = 0;
    % end

    if breakindex >= nbreakindex && abs(F_hist(itr)-F_hist(itr-1)) < 1e-6
        break
    end


end

% fprintf('err %f\n',err)
% w_subproblem = zeros(size(L,3),1);
% parfor index= 1:size(L,3)
%     % L_current = L{index};
%     L_current = L(:,:,index);
%     w_subproblem(index,1) = exp(trace(L_current'*P)/entropy_penalty);
% end
%
% w=w_subproblem/sum(w_subproblem);
% w(isnan(w))=  1/(sum(isnan(w)));
% w=w/sum(w);
%
%
% % Update Lsum
% Lsum = 0;
%
% %     parfor index = 1:size(w,2)
% %         Lsum = Lsum + L{index}*w(index);
% %     end
% parfor index = 1:size(L,3)
%     Lsum = Lsum - L(:,:,index)*w(index);
% end

% fprintf('function_value is %.8f\n', ...
%     objective_function(Lsum,U,beta))


% disp(w')


% function_val = [function_val; objective_function(L,U,sparse) ];

% if size(function_val, 1) >=2 &&...
%         abs(function_val(end,1) - function_val(end-1,1)) < 1e-8
%     breakindex = breakindex+1;
% else
%     breakindex = 0;
% end
%
% if breakindex >= 50
%     break
% end


% Output the error
% error = max(max(abs(P - U*U')));
% if error < 1e-6
% break
% end
% fprintf('error: %e\n',error);

% fprintf('The error is %f, objective function is %f\n', error, objective_function(NL2,U,beta));
UUt = U_best*U_best';
sparsity_multi = size(UUt(abs(UUt) < 1e-3),1)/(n)^2;
out.F_hist = F_hist(1:itr-1);
out.F_best = sum(L.*UUt,"all") + sparse * sum(sum(abs(UUt)));
out.itr = itr-1;
out.mu_hist = sigma_hist(1:itr-1);
end