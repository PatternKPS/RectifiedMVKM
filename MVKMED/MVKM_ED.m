%%   This function is provided (written and created) by Kristina P. Sinaga 
%%   Copyright (c) 2024 Kristina P. Sinaga 
%%   All rights reserved
%%   Contact :  krist.p.sinaga@gmail.com
%%   ------------------------------------------------------------------- %%

%%  Program for the Rectified Gaussian Kernel Multi-View K-Means Clustering 
%   Date: Oct. 20, 2023
%   We proposed A Rectified Gaussian Kernel Multi-View K-Means Clustering  
%   Abbreviated or so-called as GKMVKM, tested on Matlab R2020a
%   This is the second algorithm so-called as MVKM-ED
%-------------------------------------------------------------------------------------------------------------------
% Input   : X is a multi-view dataset (sample-view space)
%           cluster_num is the number of clusters
%           points_view is the number of data views
%           Aalpah is the exponent parameter to control the weights of V
%           Bbetah is the coefficient parameter to control the exponential
%                         distance between X and A
%           dh is the hth view number of dimensions
%-------------------------------------------------------------------------------------------------------------------
% Output : V  : the view weights
%          U  : the memberhips of all data views
%          A  : the cluster centers for h-th view data
%          index : the cluster index for multi-view data
%-------------------------------------------------------------------------------------------------------------------
function [ A, V, U, index, Param_beta] = MVKM_ED (X, cluster_num, points_view, Aalpah, Bbetah, dh)
% Replacing the parameters with some characters.
s = points_view;
c = cluster_num;
data_n = size(X{1},1);
Param_alpha = Aalpah;
Param_beta = Bbetah;
%--------------------------------------------------------------------------
% Initialize the cluster centers A
initial = randperm(data_n,c); 
for h = 1:s
    A{h} = X{h}(initial,:); % HLeaves,
end
% for h=1:s
%     [~, A{h}] = litekmeans(X{h},c); % NGs, UWA, ORL
% end
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
obj_MVKM_ED = zeros(1,max_time);   %% store the objective value 

%--------------------------------------------------------------------------
%Start the iteration
%--------------------------------------------------------------------------
while 1 && time <= max_time
%tic
fprintf('--------------  The %d Iteration Starts ----------------------\n', time); 
%% Step 1: Compute the coefficient parameter beta
% Param_beta=abs([sum(mean(X{1})*c/(time*data_n)) sum(mean(X{2})*c/(time*data_n)) sum(mean(X{3})*c/(time*data_n))]);
Param_beta=abs([sum(mean(X{1})*c/(time*data_n)) sum(mean(X{2})*c/(time*data_n)) sum(mean(X{3})*c/(time*data_n)) sum(mean(X{4})*c/(time*data_n))]);
%--------------------------------------------------------------------------
%% Step 2: Compute the memberships U
    u6=[];
    for k=1:c
        u5=[];
        for h=1:s
            u1=bsxfun(@minus,X{h},A{h}(k,:)).^2;
            u2 = exp(-Param_beta(h).*sum(u1,2));
            u3 = bsxfun(@minus,1,u2);
            u4=bsxfun(@times,u3,V(h)^Param_alpha);
            u5=[u5 u4];
        end
            u6=[u6 sum(u5,2)];
    end

    U=zeros(data_n,c);
    for i=1:data_n
        [val,idx]=min(u6(i,:));
        U(i,idx)=1;
    end
%--------------------------------------------------------------------------
%% Step 3: Update the cluster centers A
    for h=1:s
        for k=1:c 
            A_up=0;A_down=0;
            for i = 1:data_n 
                temp1=0;

                rhoexpo = bsxfun(@minus,X{h}(i,:),A{h}(k,:)).^2;
                rhoexpo_comp = exp(-Param_beta(h).*sum(rhoexpo,2));
                rhoexpo_comp2 = V(h)^(Param_alpha).*rhoexpo_comp;

                  A_up=A_up+ ( (V(h)^(Param_alpha).*rhoexpo_comp.*U(i,k)) + temp1) * X{h}(i,:); 
                  A_down=A_down+ ( (V(h)^(Param_alpha).*rhoexpo_comp.*(U(i,k))) + temp1 );                                                               
            end
            new_A{h}(k,:)=A_up./A_down; 
        end
    end  
    A = new_A;
%--------------------------------------------------------------------------
%% Step 4: Update the weighted view V
    V_Up = [];
    for h = 1 : s
        temp12 = 0;
        temp11 = 0;
        temp10 = 0;

        for k = 1 : c

            for i = 1 : data_n
                V_Up_1 = bsxfun(@minus,X{h}(i,:),A{h}(k,:)).^2;
                V_Up_2 = exp(-Param_beta(h)*sum(V_Up_1,2));
                V_Up_3 = 1 - V_Up_2;

                temp12 = temp12+(U(i,k).*V_Up_3); 

            end

            V1 = (1./temp12)^(1/(Param_alpha-1));

        end
        V2 = sum(V1);
        V_Up = [V_Up V2];
    end
    V_Down = sum(V_Up,2);

    New_V= bsxfun(@rdivide,V_Up,V_Down);     

    V = New_V;
%--------------------------------------------------------------------------
%% Computing the objective value   
for h = 1:s
    for k = 1:c
        D_exponent1 = bsxfun(@minus,X{h}(i,:),A{h}(k,:)).^2;
        D_exponent2 = exp(-Param_beta(h)*sum(V_Up_1,2));
        D_exponent3{h}(:,k) = bsxfun(@minus,1,D_exponent2);
    end
end    
obj = zeros(s,1);
for h = 1:s
    obj(h) = obj(h) + ( V(h).^Param_alpha .* sum(sum(U.*D_exponent3{h})));
end
obj_MVKM_ED(time) = sum(obj);

% fprint the result
fprintf('MVKM-ED: Iteration count = %d, MVKM-ED = %f\n', time, obj_MVKM_ED(time));
if time > 1 && (abs(obj_MVKM_ED(time) - obj_MVKM_ED(time-1)) <= 1e-4)
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
end    
