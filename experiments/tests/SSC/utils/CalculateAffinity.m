function [affinity] = CalculateAffinity(data,sigma)

% set the parameters
% sigma = 1;

for i=1:size(data,1)    
    for j=1:size(data,1)
        dist = norm(data(i,:) - data(j,:),'fro')^2;
        % dist = sqrt((data(i,1) - data(j,1))^2 + (data(i,2) - data(j,2))^2); 
        affinity(i,j) = exp(-dist/(2*sigma^2));
    end
end



