%load input data
InputDirectoryPath = uigetdir('select file path');
seeds = importdata(strcat(InputDirectoryPath,'\seeds.txt'));
%input number of clusters
k=input('enter number of clusters ');
%random numbers taken in p to take initial clusters
p = randperm(size(seeds,1));
centroids = zeros(k,size(seeds,2));
new_centroids = zeros(k,size(seeds,2));
counts = zeros(k,1);
cluster = zeros(size(seeds,1),1);
%selects random centroids
for i=1:k
centroids(i,:) = seeds(p(i),:);
end
SSE = zeros(k,1);
loop =0;
prev_SSE =0;
while 1,
    loop = loop+1;
    %calculate distances of each point to all centroids
    dist = pdist2(seeds,centroids);
    %calculate SSE
    for i=1:210
        temp = dist(i,:);
       [min_value,idx] = min(temp);
       cluster(i) = idx; %cluster assigned to input i
       SSE(idx) = SSE(idx) + pdist2(seeds(i,:),centroids(idx,:))^2;
       new_centroids(idx,:) = new_centroids(idx,:) + seeds(i,:);
       counts(idx) = counts(idx)+1;
    end
    %calculate new centroids
    for j =1:k
        new_centroids(j,:) = new_centroids(j,:) / counts(j);
    end
    centroids = new_centroids;
    counts = zeros(k,1);
    new_centroids = zeros(k,size(seeds,2));
    %break loop if 100th iteration or SSE is changed by 0.001
    if abs(prev_SSE - sum(SSE)) <=0.001 || loop ==100
        break
    end
    prev_SSE = sum(SSE);
    SSE = zeros(k,1);
end
fprintf('K= %d   SSE = %2.4f\n',k,sum(SSE));
