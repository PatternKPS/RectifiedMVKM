%%  Program for the proposed MVKM
%   Written by Kristina P. Sinaga 
%   This is the first algorithm so-called MVKM, tested on Matlab R2020a
%   Copyright (c) 2024 Kristina P. Sinaga
%-------------------------------------------------------------------------------------------------------------------
% Input   : X is a multi-view dataset (sample-view space)
%           cluster_num is the number of clusters
%           points_view is the number of data views
%           dh is the hth view number of dimensions
%-------------------------------------------------------------------------------------------------------------------
% Output : index : the cluster index for multi-view data
%-------------------------------------------------------------------------------------------------------------------
function [ index ] = mvkm (X, cluster_num, points_view, dh)
% Replacing the parameters with some characters.
s = points_view;
c = cluster_num;
data_n = size(X{1},1);
%--------------------------------------------------------------------------
% Initialize the cluster centers A
initial = randperm(data_n,c); 
% rng(10);
num_seeds = 25;
rng('shuffle');
seeds=randi(2^23-1, [1 num_seeds]);
for h = 1:s
    A{h} = X{h}(initial,:);
%     [~, A{h}] = litekmeans(X{h},c,'MaxIter', 50,'Replicates',50);
end
%--------------------------------------------------------------------------
% Initialize the weighted view V
for h =1:s
    V = ones(1,s)./ s;
end
%--------------------------------------------------------------------------
% Initializing the other variables.
time = 1;           %% iteration time
% The maximum iteration number is default 100
max_time = 100;
obj_MVKM = zeros(1,max_time);   %% store the objective value
%--------------------------------------------------------------------------
%Start the iteration
%--------------------------------------------------------------------------
while 1 && time <= max_time
%tic
fprintf('--------------  The %d Iteration Starts ----------------------\n', time); 

for h = 1:s
    for k = 1:c
        D1  = bsxfun(@minus, X{h}, A{h}(k,:));    
        D1_sum = sum((D1.^2),2);
        D11_sum{h}(:,k) = sum((D1.^2),2);
        D_ec{h}(:,k)    = bsxfun(@times, V(h).^2, D1_sum);
    end
end
   
%--------------------------------------------------------------------------
%% First step, Compute the memberships U 
U4 = [];
for k = 1:c
    U3 = [];
    
    for h = 1:s
        
        U1 = bsxfun(@minus, X{h}, A{h}(k,:));
        U1_sum = sum((U1.^2),2);
        U2 = bsxfun(@times, V(h).^2, U1_sum);
        U3 = [U3 U2];
    end
    U4 = [U4 sum(U3,2)];
end
 
U = zeros(data_n, c);
for i = 1 : data_n
    [min_dist, cluster_elem] = min(U4(i,:));
    U(i, cluster_elem) = 1;
end
%--------------------------------------------------------------------------
%% Second step: Update the cluster centers A
 
for h = 1:s
    for k = 1:c
        A{h}(k,:) = (X{h}'*U(:,k))/(sum(U(:,k)));
    end
end
%--------------------------------------------------------------------------
%% Third step: Update the weights V
 
for h = 1:s
    for k = 1:c
        D = bsxfun(@minus,X{h},A{h}(k,:));
        D2{h}(:,k) = sum(D.^2,2);
    end
    Et{h} = sum(sum(U.*D2{h}));
    Et1 = [Et{:}];
    V = (1./Et1) / (sum(1./Et1));  
end
%--------------------------------------------------------------------------
%% Computing the objective value
f1 = zeros(s,1);
for h = 1:s
    f1(h) = f1(h) + (V(h).^2.*sum(sum(U.*D_ec{h})));
end
obj_MVKM(time) = sum(f1);

% fprint the result 
fprintf('MVKM: Iteration count = %d, MVKM = %f\n', time, obj_MVKM(time));
if time > 1 && (abs(obj_MVKM(time) - obj_MVKM(time-1)) <= 1e-4)
   index = [];
   for i = 1:data_n
       [num idx]=max(U(i,:));
       index=[index;idx];
   end
%    [ index ] = Utoindex (U, data_n, s, c, V);
   fprintf('------------ The Iteration has finished.-----------\n\n');
   break;
end
time = time +1;

end