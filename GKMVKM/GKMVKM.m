%%   This code is provided (written and created) by Kristina P. Sinaga 
%%   Copyright (c) 2024 Kristina P. Sinaga 
%%   All rights reserved
%%   Contact :  krist.p.sinaga@gmail.com
%%   ------------------------------------------------------------------- %%

%%  Program for the Rectified Gaussian Kernel Multi-View K-Means Clustering 
%   Date: Oct. 20, 2023
%   We proposed A Rectified Gaussian Kernel Multi-View K-Means Clustering  
%   Abbreviated or so-called as GKMVKM, tested on Matlab R2020a
%   This is the third algorithm so-called as GKMVKM
%-------------------------------------------------------------------------------------------------------------------
% Input   : X is a multi-view dataset (sample-view space) 
%           cluster_num is the number of clusters
%           points_view is the number of data views
%           Aalpah is the exponent parameter to control the weights of V
%           Bbetah is the coefficient parameter to control the exponential
%                         distance between X and A
%           p is the parameter to handle the performance of the exp dist
%           dh is the hth view number of dimensions
%-------------------------------------------------------------------------------------------------------------------
% Output : V  : the view weights
%          U  : the memberhips of all data views
%          A  : the cluster centers for h-th view data
%          index : the cluster index for multi-view data
%-------------------------------------------------------------------------------------------------------------------
function [ A, V, U, index, Param_beta] = GKMVKM (X, cluster_num, points_view, Aalpah, Bbetah, p, dh)
% Replacing the parameters with some characters.
s = points_view;
c = cluster_num;
data_n = size(X{1},1);
Param_alpha = Aalpah;
Param_beta = Bbetah;
Param_P = p;
%% Compute P using estimator
%% Computing the objective value   
% for h = 1:s
%     for k = 1:c
%         ParamP_1 = bsxfun(@minus,X{h}(i,:),A{h}(k,:)).^2;
%         D_exponent2 = (-Param_beta(h)*sum(D_exponent1,2));
%         D_exponent3{h} = exp(D_exponent2);
%     end
% end
%--------------------------------------------------------------------------
% Initialize the cluster centers A
initial = randperm(data_n,c); 
for h = 1:s
    A{h} = X{h}(initial,:); % HLeaves, UWA, NG
end
% for h=1:s
%     [~, A{h}] = litekmeans(X{h},c); % ORL, NUS
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
obj_GKMVKM = zeros(1,max_time);   %% store the objective value 

%--------------------------------------------------------------------------
%Start the iteration
%--------------------------------------------------------------------------
while 1 && time <= max_time
%tic
fprintf('--------------  The %d Iteration Starts ----------------------\n', time); 

%--------------------------------------------------------------------------
%% First step, Compute the coefficient parameter beta
Param_beta = [abs([sum(mean(X{1})*c/(time*data_n)) sum(mean(X{2})*c/(time*data_n))]);];
% Param_beta = [abs([sum(mean(X{1})*c/(time*data_n)) sum(mean(X{2})*c/(time*data_n)) sum(mean(X{3})*c/(time*data_n))]);];
%--------------------------------------------------------------------------
%% 2nd step, Compute the memberships U
    u6=[];
    for k=1:c
        u5=[];
        for h=1:s
            u1=bsxfun(@minus,X{h},A{h}(k,:)).^2;
            u11 = (-Param_beta(h).*sum(u1,2));
            u12 = exp(u11).^Param_P;
            u3 = bsxfun(@minus,1,u12);
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
%% 3rd step, Update the view weights V
    V_Up = [];
    for h = 1 : s
        temp12 = 0;
        temp11 = 0;
        temp10 = 0;

        for k = 1 : c

            for i = 1 : data_n
                V_Up_1 = bsxfun(@minus,X{h}(i,:),A{h}(k,:)).^2;
                V_Up_12 = (-Param_beta(h)*sum(V_Up_1,2));
                V_Up_13 = bsxfun(@minus,1,(exp(V_Up_12)).^Param_P);
                
                V_Up_3 = U(i,k).*V_Up_13;
                temp12 = temp12+V_Up_3; 

            end
            V1 = (1./temp12)^(1/(Param_alpha-1));
        end
        V2 = sum(V1);
        V_Up = [V_Up V2];
    end
    V_Down = sum(V_Up,2);
    New_V  = bsxfun(@rdivide,V_Up,V_Down);     
    V      = New_V;

%--------------------------------------------------------------------------
%% 4th step, Update the cluster centers A
    for h=1:s
        for k=1:c
            A_up=0;A_down=0;
            for i = 1:data_n 
                temp1=0;

                rhoexpo = bsxfun(@minus,X{h}(i,:),A{h}(k,:)).^2;
                rhoexpo1 = (-Param_beta(h).*sum(rhoexpo,2));
                rhoexpo2 = (exp(rhoexpo1)).^Param_P;
                A_up=A_up+ ( (V(h)^(Param_alpha).*rhoexpo2.*U(i,k)) + temp1) * X{h}(i,:); % Size X{r}(j,:) =>> [1 47]
                A_down=A_down+ ( (V(h)^(Param_alpha).*rhoexpo2.*(U(i,k))) + temp1 );                                                               
            end
            new_A{h}(k,:)=A_up./A_down; 
        end
    end  
    A = new_A;
%--------------------------------------------------------------------------
%% Computing the objective value   
for h = 1:s
    for i = 1: data_n 
        for k = 1:c
            D_exponent1 = bsxfun(@minus,X{h}(i,:),A{h}(k,:)).^2;
            D_exponent2 = (-Param_beta(h)*sum(D_exponent1,2));
            D_exponent3{h}(:,k) = bsxfun(@minus,1,(exp(D_exponent2)).^Param_P);
        end
    end
end

obj = zeros(s,1);
for h = 1:s
    obj(h) = obj(h) + ( V(h).^Param_alpha .* sum(sum(U.*D_exponent3{h})));
end
obj_GKMVKM(time) = sum(obj);

% fprint the result
fprintf('GKMVKM: Iteration count = %d, GKMVKM = %f\n', time, obj_GKMVKM(time));
if time > 1 && (abs(obj_GKMVKM(time) - obj_GKMVKM(time-1)) <= 1e-4)
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
