% ALGORITHM: LMPFE
% <multi/many> <real/integer/label/binary/permutation>
% Evolutionary algorithm with local model based Pareto front estimation
% fPFE  --- 0.1 --- Frequency of employing generic front modeling
% K     --- 5  --- Number of subregions

%------------------------------- Reference --------------------------------
% Y. Tian, L. Si, X. Zhang, K. C. Tan, and Y. Jin, Local model based Pareto
% front estimation for multi-objective optimization, IEEE Transactions on
% Systems, Man, and Cybernetics: Systems, 2023, 53(1): 623-634.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
clc
clear
close all

% Problem settings
PortPosition = [0 50];
MaxRobotNo = 3;
global XWind YWind RWind
XWind = 69.73;
YWind = 58.15;
RWind = 15;
allpop = [];
load('x.mat')
load('y.mat')
load('r.mat')
XVessel = x(1:10);
YVessel = y(1:10);
RVessel = r(1:10);
DistanceWeight = zeros(1,length(XVessel));
for i = 1:length(XWind)
    distance = sqrt((XVessel-XWind(i)).^2+(YVessel-YWind(i)).^2);
    DistanceWeight(find(distance<RWind(i))) = DistanceWeight(find(distance<RWind(i))) + distance(find(distance<RWind(i)))./RWind(i);
end
GenotypeLength = [4 4 2 2];
VarietyNumber = length(XVessel);
nPop=100;
MaxIt=100;
FE = 0;

 %% Parameter setting
 fPFE = 0.1;
 K = 5;
%% Generate random population
Population = randi([0 1],nPop,VarietyNumber*sum(GenotypeLength));
Populationobjs = DecodeFunction(Population,XVessel,YVessel,RVessel,GenotypeLength,size(Population,1),VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
Populationobjs = Populationobjs(:,161:163);
FrontNo    = NDSort(Populationobjs,inf);
 %% Generate K subregions
 [Center,R] = adaptiveDivision(Populationobjs,K);
 %% Calculate the intersection point on each subregion
 % Initialze parameter P
 P = ones(K,3);
  % Calculate the fitness of each solution
  [App,Dis] = subFitness(Populationobjs,P,Center,R);
  Dis       = sort(Dis,2);
  Crowd     = Dis(:,1) + 0.1*Dis(:,2);
  theta     = 0.8;
  preApp    = mean(App);
  preCrowd  = mean(Crowd);
 
while FE <= 10000
     % Mating
     MatingPool = TournamentSelection(2,nPop,FrontNo,-theta*Crowd-(1-theta)*App);
     Population = Population(MatingPool,:);
     Offspring = [];
     for i = 1:nPop/2
         % Choose parents
         Pcr = randperm(nPop);
         % Generate an offspring
         p1 = Population(Pcr(1),:);
         p2 = Population(Pcr(2),:);
         pos1 = randi([1,VarietyNumber*sum(GenotypeLength)],1,1);
         pos2 = randi([1,VarietyNumber*sum(GenotypeLength)],1,1);
         if pos1 > pos2
             temp = pos1;
             pos1 = pos2;
             pos2 = temp;
         end
         temp = p1(pos1:pos2);
         p1(pos1:pos2) = p2(pos1:pos2);
         p2(pos1:pos2) = temp;
         pos = randi([1,VarietyNumber*sum(GenotypeLength)],1,1);
         p1(pos) = 1-p1(pos);
         pos = randi([1,VarietyNumber*sum(GenotypeLength)],1,1);
         p2(pos) = 1-p2(pos);
         
         Offspring = [Offspring; p1];
         Offspring = [Offspring; p2];
     end

     FE = FE+size(Offspring,1)
     % Generic front modeling
     if ~mod(ceil(FE/nPop),ceil(fPFE*ceil(10000/nPop))) || fPFE == 0
         % Update subregions
         [Center,R] = adaptiveDivision(Populationobjs,K);
         % PF modeling
         P = subGFM(Populationobjs,Center,R,FrontNo);
     end
     Offspringobjs = DecodeFunction(Offspring,XVessel,YVessel,RVessel,GenotypeLength,size(Offspring,1),VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
     Offspringobjs = Offspringobjs(:,161:163);
     
     Population = [Population; Offspring];
     Populationobjs = [Populationobjs; Offspringobjs];
     
    %% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(Populationobjs,nPop);
    Next             = find(FrontNo<=MaxFNo);
      %% Environmental selection
    [App,Dis] = subFitness(Populationobjs(Next,:),P,Center,R);
    Choose    = LastSelection(Populationobjs(Next,:),FrontNo(Next),App,Dis,theta,nPop);

     %% Population for next generation
    Population = Population(Next(Choose),:);
    Populationobjs = DecodeFunction(Population,XVessel,YVessel,RVessel,GenotypeLength,size(Population,1),VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
    Populationobjs = Populationobjs(:,161:163);
    FrontNo    = FrontNo(Next(Choose));
    App        = App(Choose);
    Dis        = sort(Dis(Choose,Choose),2);
    Crowd      = Dis(:,1) + 0.1*Dis(:,2);
    % Update theta
    [theta,preApp,preCrowd] = UpdateTheta(preApp,preCrowd,App,Crowd);

end
objs = DecodeFunction(Population,XVessel,YVessel,RVessel,GenotypeLength,size(Population,1),VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
Epa = objs(:,161:163);
R = [520 260 2.5];
hypervolume(Epa, R, 100000)
IsDomi = IDAf(Epa);
DomiIndex = find(IsDomi==1);
epa = Epa(DomiIndex,:);
function Choose = LastSelection(PopObj,FrontNo,App,Dis,theta,N)
% Select part of the solutions in the last front

    %% Identify the extreme solutions
    NDS = find(FrontNo==1);
    [~,Extreme] = min(repmat(sqrt(sum(PopObj(NDS,:).^2,2)),1,size(PopObj,2)).*sqrt(1-(1-pdist2(PopObj(NDS,:),eye(size(PopObj,2)),'cosine')).^2),[],1);
    nonExtreme  = ~ismember(1:length(FrontNo),NDS(Extreme));
    %% Environmental selection
    Last   = FrontNo == max(FrontNo);
    Choose = true(1,size(PopObj,1));
    
    %% Non-dominated sort convergence and diversity
    while sum(Choose) > N
        Remain    = find(Choose&Last&nonExtreme);
        dis       = sort(Dis(Remain,Choose),2);
        dis       = dis(:,1) + 0.1*dis(:,2);
        fitness   = theta*dis + (1-theta)*App(Remain);
        [~,worst] = min(fitness);
        Choose(Remain(worst)) = false;
    end
end
function [App,Dis] = subFitness(PopObj,P,Center,R)
% Update intersection point solution in each subregion

%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    K          = length(R);
    [N,M]      = size(PopObj);

    % Normalize the population
    fmin       = min(PopObj,[],1);
    fmax       = max(PopObj,[],1);
    Obj        = (PopObj-repmat(fmin,N,1))./repmat(fmax-fmin,N,1);
    
    %% Calculate intersection point in each subregion
    if K == 1
        InterPoint = interPoint(Obj,P);
    else
        InterPoint = ones(N,M);
        
        % Allocation
        transformation = Allocation(Obj,Center,R);
       
        for i = 1 : K
            current     = find(transformation == i);
            if ~isempty(current)
                sInterPoint = interPoint(Obj(current,:),P(i,:));
                InterPoint(current,:) = sInterPoint;
            end
        end     
    end
    
    % Calculate the diversity and convergence of intersection points
    App = min(InterPoint-Obj,[],2);
%     App = sqrt(sum(InterPoint.^2,2)) - sqrt(sum(Obj.^2,2));

    % Calculate the diversity of each solution 
%     Dis = pdist2(InterPoint,InterPoint);
    Dis = distMax(InterPoint);
    Dis(logical(eye(length(Dis)))) = inf; 
end

function InterPoint = interPoint(PopObj,P)
% Calcualte the approximation degree of each solution, and the distances
% between the intersection points of the solutions

    [N,~] = size(PopObj);
    
    %% Calculate the intersections by gradient descent
    P     = repmat(P,N,1);      % Powers
    r     = ones(N,1);          % Parameters to be optimized
    lamda = zeros(N,1) + 0.002;   % Learning rates
    E     = sum((r.*PopObj).^P,2) - 1;   % errors
    for i = 1 : 1000
        newr = r - lamda.*E.*sum(P.*PopObj.^P.*r.^(P-1),2);
        newE = sum((newr.*PopObj).^P,2) - 1;
        update         = newr > 0 &sum(newE.^2) < sum(E.^2);
        r(update)      = newr(update);
        E(update)      = newE(update);
        lamda(update)  = lamda(update)*1.002; 
        lamda(~update) = lamda(~update)/1.002;
    end
    InterPoint = PopObj.*r;
end

function Dis = distMax(X)
% distMax pairwise distance between one set of observations
% Dis = distMax(X) returns a matrix D containing the maximum absolute
%   distance per dimension between each pair of observations in the MX-by-N
%   data matrix X and MX-by-N data matrix X. 

%   Example:
%      X = randn(100, 5);
%      D = distMax(X,Y);
%   >>size(D) = 100*100

    if isempty(X)
        error('X must be a non-empty matrix');
    end

    [N,~] = size(X); % nx,p
    Dis = zeros(N,N);
    for i = 1 : N
        for j = i+1 : N
            Dis(i,j) = max(abs(X(i,:)-X(j,:)));
        end
    end
    Dis = Dis + Dis';
end