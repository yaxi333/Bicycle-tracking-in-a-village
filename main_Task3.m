%% main_Task3

clear all;clc;

% a
Xk = MapProblemGetPoint();

% b 

% Time steps
K = size(Xk,2);
% Dimension of States
n = size(Xk,1); 
% Sample time
T = 1;
% CV motion model
A = [1 0 T 0;0 1 0 T; 0 0 1 0; 0 0 0 1];
% As equation 9 shows
H = [0 0 1 0; 0 0 0 1];
% PF-Resample
proc_f = @(X_kmin1) (f(X_kmin1, A));
meas_h = @(X_k) (h(X_k, H));
plotFunc = @(xfp)(plotParticle(xfp));

% Resample
bResample = true;
% Particle Numbers 
N = 20000;  
% Tuned process noise 
Q = diag([0 0 1 1]);
% Tuned measurement noise 
sigmar = 0.1;
R = diag([sigmar sigmar].^2);

% Generate states and measurements
Xv = [0;0];
for k = 2:K
    rk = mvnrnd(zeros(n,1),R).';
    % True velocity
    Xv = [Xv Xk(:,k)-Xk(:,k-1)]; 
    % Velocity measurement
    yk(:,k-1) = Xv(:,k) + rk;
end
% Generate the state matrix
Xs = [Xk;Xv]; 
Y = yk;

% Initialize the prior
x_0 = Xs(:,1);
P_0 = zeros(4);


task = 'e';

switch task
    
    case 'd'
        
        % Assume that the initial prior is known
        [xfp, Pfp, Xp, Wp] = pfFilter(x_0, P_0, Y, proc_f, Q, meas_h, R, N, bResample, plotFunc);
        % Calculate the averaged Mean Squared error of the x and y position
        err = mean((xfp(1:2,:) - Xs(1:2,2:end)).^2,2);
        disp('The averaged Mean Squared error of the x and y position are:');
        disp(err);
        
    case 'e'               
                         
        % Assume that the initial prior is unknown
        prior = false;
        [xfp, Pfp, Xp, Wp] = pfFilter(x_0, P_0, Y, proc_f, Q, meas_h, R, N, bResample, plotFunc, prior);
        % Calculate the averaged Mean Squared error of the x and y position
        err = mean((xfp(1:2,:) - Xs(1:2,2:end)).^2,2);
        disp('The averaged Mean Squared error of the x and y position are:');
        disp(err);
end


% Functions
 function plotParticle(xfp)
    hold on
    plot(xfp(1,:),xfp(2,:),'g*-','LineWidth',2)
 end     
 
function [xfp, Pfp, Xp, Wp] = pfFilter(x_0, P_0, Y, proc_f, proc_Q, meas_h, meas_R, N, bResample, plotFunc, prior)
%PFFILTER Filters measurements Y using the SIS or SIR algorithms and a
% state-space model.
%
% Input:
%   x_0         [n x 1] Prior mean
%   P_0         [n x n] Prior covariance
%   Y           [m x K] Measurement sequence to be filtered
%   proc_f      Handle for process function f(x_k-1)
%   proc_Q      [n x n] process noise covariance
%   meas_h      Handle for measurement model function h(x_k)
%   meas_R      [m x m] measurement noise covariance
%   N           Number of particles
%   bResample   boolean false - no resampling, true - resampling
%   plotFunc    Handle for plot function that is called when a filter
%               recursion has finished.
% Output:
%   xfp         [n x K] Posterior means of particle filter
%   Pfp         [n x n x K] Posterior error covariances of particle filter
%   Xp          [n x N x K] Non-resampled Particles for posterior state distribution in times 1:K
%   Wp          [N x K] Non-resampled weights for posterior state x in times 1:K

% Your code here, please. 
% If you want to be a bit fancy, then only store and output the particles if the function
% is called with more than 2 output arguments.

n = size(x_0,1);% dimensions of the state
m = size(Y,1);% dimensions of the measurement
K = size(Y,2);% the time

Xp = zeros(n,N,K);Wp = zeros(N,K);
xfp = zeros(n,K);Pfp = zeros(n,n,K);

switch bResample
    
    
    case false
        
         if nargin < 11
           % Calculate the particle X0 for time k =0
           X_kmin1= (mvnrnd(x_0,P_0,N)).';
         else
           % Generate the Particles which follow the uniform distribution
           % in the map
            axis_xy = [10;8];
            X_kmin1 = [rand(2,N).*axis_xy + [1;1];zeros(2,N)];
        end
        W_kmin1 = (1/N).*ones(1,N);% initial prior weight is equally divided

        % Calculate the particle Xi for time k >= 2
        for k = 1:K
            [X_kmin1, W_kmin1] = pfFilterStep(X_kmin1, W_kmin1, Y(:,k), proc_f, proc_Q, meas_h, meas_R);
            W_kmin1 = W_kmin1.*isOnRoad(X_kmin1(1,:),X_kmin1(2,:))';
            Wk = W_kmin1./sum(W_kmin1);
            Xk = X_kmin1;
             % Calculate the Posterior means of particle filter  
             xfp(:,k) = sum(Wk.*Xk,2);
             % Calculate the Posterior error covariances of particle filter
             Pfp(:,:,k) = (Wk.*(Xk - xfp(:,k)))*((Xk - xfp(:,k))');  
             Xp(:,:,k) = Xk;
             Wp(:,k) = Wk.';
             
             % Update the next time prior
             X_kmin1 = Xk;
             W_kmin1 = Wk; 
             
        end
        hold on;
        plotFunc(xfp);

        
    case true
        
        if nargin < 11
           % Calculate the particle X0 for time k =0
           X_kmin1= (mvnrnd(x_0,P_0,N)).';
        else
            axis_xy = [10;8];
            X_kmin1 = [rand(2,N).*axis_xy + [1;1];zeros(2,N)];
        end
        W_kmin1 = (1/N).*ones(1,N);% initial prior weight is equally divided

        % Calculate the particle Xi for time k >= 2
        for k = 1:K
            [X_kmin1, W_kmin1] = pfFilterStep(X_kmin1, W_kmin1, Y(:,k), proc_f, proc_Q, meas_h, meas_R);
            W_kmin1 = W_kmin1.*isOnRoad(X_kmin1(1,:),X_kmin1(2,:))';
            W_kmin1 = W_kmin1./sum(W_kmin1);

            % Resample
            [Xk, Wk, j] = resampl(X_kmin1, W_kmin1);
            
             % Calculate the Posterior means of particle filter  
             xfp(:,k) = sum(Wk.*Xk,2);
             % Calculate the Posterior error covariances of particle filter
             Pfp(:,:,k) = (Wk.*(Xk - xfp(:,k)))*((Xk - xfp(:,k))');  
             Xp(:,:,k) = Xk;
             Wp(:,k) = Wk.';
             
             % Update the next time prior
             X_kmin1 = Xk;
             W_kmin1 = Wk;             
        end
        hold on;
        plotFunc(xfp);
end
end

function [Xr, Wr, j] = resampl(X, W)
%RESAMPLE Resample particles and output new particles and weights.
% resampled particles. 
%
%   if old particle vector is x, new particles x_new is computed as x(:,j)
%
% Input:
%   X   [n x N] Particles, each column is a particle.
%   W   [1 x N] Weights, corresponding to the samples
%
% Output:
%   Xr  [n x N] Resampled particles, each corresponding to some particle 
%               from old weights.
%   Wr  [1 x N] New weights for the resampled particles.
%   j   [1 x N] vector of indices refering to vector of old particles

% Your code here!
n = size(X,1);N = size(X,2);
% Initialize the matrix
Xr = zeros(n,N);j = zeros(1,N);
% Normalise the weights 
W = W/(sum(W));
% Calculate the cumulative sum of the weights matrix
Csum = cumsum(W);
% New weights for the resampled particles is equal
Wr = ones(1,N)./N; 
% Using the uniform distribution to distribute the weights again
r = rand(1,N);
r = sort(r,'descend');

for i = 1:N
    for k = 1:N % Search in the old weight matrix to find the satisfied weight interval
        if Csum(k) >= r(i)
            Xr(:,i) = X(:,k);
            j(i) = k;
            break
        end
    end
end
end



function [X_k, W_k] = pfFilterStep(X_kmin1, W_kmin1, yk, proc_f, proc_Q, meas_h, meas_R)
n = size(X_kmin1,1);
N = size(X_kmin1,2);
m = size(yk,1);

% initialize the matrix
X_k = zeros(n,N);
W_k = zeros(1,N);
H_k= [];

% Calculate Particles for state x in time k
fk = proc_f(X_kmin1);
X_k= (mvnrnd(fk.',proc_Q)).';

% Calculate the probability of the given y_k according to the Gaussian density p(y_k|x_k) 
hk = meas_h(X_k);
H_k = (mvnpdf(yk.', hk.',meas_R)).';

% Calculate Weights for state x in time k
W_k = W_kmin1.*H_k;

% Normalization
W_k = W_k./sum(W_k);
end


function [u] = isOnRoad(x,y)
% Input:    vectors with x and y positions
%
% Output:   a vector u such that u(i) = 1 if (x(i),y(i)) is on the road
%           and 0 otherwise. 
%


%   Make sure that x and y are column vectors
n   =   length(x);      
x = reshape(x,n,1); 
y = reshape(y,n,1);

%   The number of buildings (including two rectangles in the middle)
m = 9;             

%   To check if any vector is in any building we create
%   matrices of size n x m:
X = x*ones(1,m);
Y = y*ones(1,m);
%   We should check that we are on the map
bounds = ([1+i 1+9*i 11+9*i 11+i]);

%   And that we are not in any of these houses
house = zeros(m,5);
house(1,:) = ([2+5.2*i 2+8.3*i 4+8.3*i 4+5.2*i 2+5.2*i]);%House 1
house(2,:) = ([2+3.7*i 2+4.4*i 4+4.4*i 4+3.7*i 2+3.7*i]);%House 2
house(3,:) = ([2+2*i 2+3.2*i 4+3.2*i 4+2*i 2+2*i]);%House 3
house(4,:) = ([5+i 5+2.2*i 7+2.2*i 7+i 5+i]);%House 4
house(5,:) = ([5+2.8*i 5+5.5*i 7+5.5*i 7+2.8*i 5+2.8*i]);%House 5
house(6,:) = ([5+6.2*i 5+9*i 7+9*i 7+6.2*i 5+6.2*i]);%House 6
house(7,:) = ([8+4.6*i 8+8.4*i 10+8.4*i 10+4.6*i 8+4.6*i]);%House 7
house(8,:) = ([8+2.4*i 8+4*i 10+4*i 10+2.4*i 8+2.4*i]);%House 8
house(9,:) = ([8+1.7*i 8+1.8*i 10+1.8*i 10+1.7*i 8+1.7*i]);%House 9

%   Let us check if we are in any of the houses:
X1 = X >= ones(n,1)*real(house(:,1))';
X2 = X <= ones(n,1)*real(house(:,3))';
Y1 = Y >= ones(n,1)*imag(house(:,1))';
Y2 = Y <= ones(n,1)*imag(house(:,2))';
XX = X1.*X2;               % Finds houses that match the x-vector
YY = Y1.*Y2;               % Finds houses that match the y-vector
UU = XX.*YY;               % Finds houses that match both x and y
u1 = 1-min(1,(sum(UU')))'; % Sets u(i)=0 if (x(i),y(i)) is in a house

%   We should also make sure that the vectors are in the village
x3 = x > ones(n,1)*real(bounds(1))';
x4 = x < ones(n,1)*real(bounds(3))';
y3 = y > ones(n,1)*imag(bounds(1))';
y4 = y < ones(n,1)*imag(bounds(2))';

xx = x3.*x4;        %   Checks that the x-coordinates are in the village
yy = y3.*y4;        %   and that the y-coordinates are in the village
u2 = xx.*yy;        %   Both must be inside

% Finally, we set the output to zero if (x,y) is either in a building
% or outside the village:
u = u1.*u2;

end
                      

function X_k = f(X_kmin1, A)
%
% X_kmin1:  [n x N] N states vectors at time k-1
% A:        [n x n] Matrix such that x_k = A*x_k-1 + q_k-1
    X_k = A*X_kmin1;
end

function H_k = h(X_k, H)
%
% X_k:  [n x N] N states
% H:    [m x n] Matrix such that y = H*x + r_k
    H_k = H*X_k;
end

function Xk = MapProblemGetPoint
clear all
%This file draw a map of the village, and allows us to manually
%draw the trajectory of the vehicle.

figure(1)
clf
hold on
plot([1+i 1+9*i 5+9*i])
plot([7+9*i 11+9*i 11+i 7+i]);plot([5+i 1+i])
plot([2+5.2*i 2+8.3*i 4+8.3*i 4+5.2*i 2+5.2*i])%House 1
plot([2+3.7*i 2+4.4*i 4+4.4*i 4+3.7*i 2+3.7*i])%House 2
plot([2+2*i 2+3.2*i 4+3.2*i 4+2*i 2+2*i])%House 3
plot([5+i 5+2.2*i 7+2.2*i 7+i])%House 4
plot([5+2.8*i 5+5.5*i 7+5.5*i 7+2.8*i 5+2.8*i])%House 5
plot([5+6.2*i 5+9*i]);plot([7+9*i 7+6.2*i 5+6.2*i])%House 6
plot([8+4.6*i 8+8.4*i 10+8.4*i 10+4.6*i 8+4.6*i])%House 7
plot([8+2.4*i 8+4*i 10+4*i 10+2.4*i 8+2.4*i])%House 8
plot([8+1.7*i 8+1.8*i 10+1.8*i 10+1.7*i 8+1.7*i])%House 9

axis([0.8 11.2 0.8 9.2])
title('A map of the village','FontSize',20)

disp('Start clicking in the village to create a trajectory!')
disp('Press "Return" to finish.')

load("Xk.mat");

% [X,Y]=ginput;
plot([X+Y*i],'-*')

Xk = [X';Y']
save Xk
end




