function NL2 = generate_laplacian_matrix(data,sigma)

affinity = CalculateAffinity(data,sigma);
% figure,imshow(affinity,[]), title('Affinity Matrix'),truesize([400 400])

D = zeros(size(affinity));

% compute the degree matrix
for i=1:size(affinity,1)
    D(i,i) = sum(affinity(i,:));
end

% compute the normalized laplacian / affinity matrix (method 1)
%NL1 = D^(-1/2) .* L .* D^(-1/2);
for i=1:size(affinity,1)
    for j=1:size(affinity,2)
        NL1(i,j) = affinity(i,j) / (sqrt(D(i,i)) * sqrt(D(j,j)));  
    end
end

% compute the normalized laplacian (method 2)  eye command is used to
% obtain the identity matrix of size m x n

 NL2 = eye(size(affinity,1),size(affinity,2)) - (D^(-1/2) * affinity * D^(-1/2));

end
