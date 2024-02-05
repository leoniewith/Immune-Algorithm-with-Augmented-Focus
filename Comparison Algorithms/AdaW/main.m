% ALGORITHM: AdaW
% <multi/many> <real/integer/label/binary/permutation>
% Evolutionary algorithm with adaptive weights

%------------------------------- Reference --------------------------------
% M. Li and X. Yao, What weights work for you? Adapting weights for any
% Pareto front shape in decomposition-based evolutionary multiobjective
% optimisation, Evolutionary Computation, 2020, 28(2): 227-253.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

clc
close all
clear

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
FE=0;

GenotypeLength = [4 4 2 2];
VarietyNumber = length(XVessel);

nPop=100;
MaxIt=100;%

%% Generate the weight vectors
[W,nPop] = UniformPoint(nPop,3);  % Generate the weight vectors
T = ceil(nPop/10);

%% Detect the neighbours of each weight
B = pdist2(W,W);
[~,B] = sort(B,2);
B = B(:,1:T);

%% Generate random population
Population = randi([0 1],nPop,VarietyNumber*sum(GenotypeLength));
objs = DecodeFunction(Population,XVessel,YVessel,RVessel,GenotypeLength,nPop,VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
objs = objs(:,161:163);
Z = min(objs,[],1);

%% Generate an archive set
Archive = Population(NDSort(objs,1)==1, :);
Archive_temp = Population;
iter = 1;
while iter <= MaxIt
    for i = 1 : nPop
        % Choose parents
        if rand < 0.9
            P = B(i,randperm(size(B,2)));
        else
            P = randperm(nPop);
        end
        % Generate an offspring
        p1 = Population(P(1),:);
        p2 = Population(P(2),:);
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
        
        Offspring = p1;
        obj = DecodeFunction(Offspring,XVessel,YVessel,RVessel,GenotypeLength,1,VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
        Archive_temp(i,:) = Offspring;
        
        % Update the ideal point
        Z = min(Z ,obj(161:163));
        % Pick a neighbour to update
        objs = DecodeFunction(Population,XVessel,YVessel,RVessel,GenotypeLength,size(Population,1),VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
        objs = objs(:,161:163);
        
        g_old = max(abs(objs(P,:)-repmat(Z,size(P,1),1))./W(P,:),[],2);
        g_new = max(repmat(abs(obj(161:163)-Z),size(P,1),1)./W(P,:),[],2);
        if ~isempty(P(find(g_old >= g_new,1)))
            Population(P(find(g_old >= g_new,1)),:) = Offspring;
        end
    end
    FE = FE + nPop;
    Archive = [Archive; Archive_temp];
    % Maintenance operation in the archive set
    [as, ~] = size(Archive);
    objs = DecodeFunction(Archive,XVessel,YVessel,RVessel,GenotypeLength,as,VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
    objs = objs(:,161:163);
    Archive = Archive(NDSort(objs,1)==1,:);
    Archive = ArchiveUpdate(Archive, 2 * nPop);
    % Update weights
    if ~mod(ceil(FE/nPop),ceil(0.05*ceil(10000/nPop))) && FE <= 10000*0.9
        %% Routine to find undeveloped individuals (correspondingly their weights) in the archive set
        % Normalisation
        [N_arc,~]     = size(Archive);
        
        [as, ~] = size(Archive);
        objs = DecodeFunction(Archive,XVessel,YVessel,RVessel,GenotypeLength,as,VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
        objs = objs(:,161:163);
        
        fmin_arc      = min(objs);
        fmax_arc      = max(objs);
        Archiveobjs   = (objs - repmat(fmin_arc,N_arc,1) )./repmat(fmax_arc - fmin_arc,N_arc,1);
        
        [as, ~] = size(Population);
        popobjs = DecodeFunction(Population,XVessel,YVessel,RVessel,GenotypeLength,as,VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
        popobjs = popobjs(:,161:163);
        
        Populaionobjs = (popobjs - repmat(fmin_arc,nPop,1) )./repmat(fmax_arc - fmin_arc,nPop,1);
        % Euclidean distance between individuals in the archive set and individuals in the Population
        dis1 = pdist2(Archiveobjs,Populaionobjs);
        dis1 = sort(dis1,2);
        % Euclidean distance between any two individuals in the archive set
        dis2 = pdist2(Archiveobjs,Archiveobjs);
        dis2 = sort(dis2,2);
        % Calculate the niche size(median of the distances from their closest solution in the archive )
        niche_size = median(dis2(:,2));
        % Find undeveloped
        Archive_und = Archive(dis1(:,1) >= niche_size, :);
        [N_und, ~] = size(Archive_und);
        
        %% If the undeveloped individuals are promising then add them into the evolutionary Population
        % Obtain their corresponding weights.
        if ~isempty(Archive_und)
            [as, ~] = size(Archive_und);
            objs = DecodeFunction(Archive_und,XVessel,YVessel,RVessel,GenotypeLength,as,VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
            Archive_undobjs = objs(:,161:163);
            W1 = (Archive_undobjs - repmat(Z,N_und,1))./repmat( sum(Archive_undobjs,2)-repmat(sum(Z),N_und,1), 1, 3);
            for i = 1 : size(W1,1)
                W_all = [W;W1(i,:)];
                B1 = pdist2(W_all,W_all);
                B1(logical(eye(length(B1)))) = inf;
                [~,B1] = sort(B1,2);
                B1 = B1(:,1:T);
                
                Population1 = [Population; Archive_und(i,:)];
                Population2 = Population1(B1(end,:),:);
                
                [as, ~] = size(Population2);
                objs = DecodeFunction(Population2,XVessel,YVessel,RVessel,GenotypeLength,as,VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
                Population2objs = objs(:,161:163);
                
                Value_Tche_all = max(abs(Population2objs-repmat(Z,T,1))./repmat(W1(i,:),T,1),[],2);
                
                [as, ~] = size(Archive_und(i,:));
                objs = DecodeFunction(Archive_und(i,:),XVessel,YVessel,RVessel,GenotypeLength,as,VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
                Archive_undobjs = objs(:,161:163);
                
                Value_Tche     = max(abs(Archive_undobjs -    Z     )./W1(i,:),[],2);
                index = find(Value_Tche_all<Value_Tche, 1);
                
                if isempty(index)
                    % Put the wight into the W, as well as the corresponding solution
                    W = [W;W1(i,:)];
                    Population = [Population; Archive_und(i,:)];
                    
                    % Update neighbour solutions after adding a weight
                    P = B1(end,:);
                    
                    [as, ~] = size(Population);
                    objs = DecodeFunction(Population,XVessel,YVessel,RVessel,GenotypeLength,as,VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
                    Populationobjs = objs(:,161:163);
                    
                    
                    g_old = max( abs( Populationobjs(P,:) - repmat(Z,T,1) )./W(P,:),[],2 );
                    g_new = max( abs( repmat(Archive_undobjs,T,1) - repmat(Z,T,1) )./W(P,:),[],2 );
                    tempg = P(g_old > g_new);
                    for jjj = 1:length(tempg)
                        Population(tempg(jjj), :) = Archive_und(i,:);
                    end
                end
            end
        end
        
        %% Delet the poorly performed weights until the size of W is reduced to N
        % find out the solution that is shared by the most weights in the population
        while size(Population,1) > nPop
            
            [as, ~] = size(Population);
            objs = DecodeFunction(Population,XVessel,YVessel,RVessel,GenotypeLength,as,VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
            Populationobjs = objs(:,161:163);
            
            [~,ai,bi] = unique(Populationobjs,'rows');
            if length(ai) == length(bi)   % If every solution in the population corresponds to only one weight
                % Normalisation
                fmax  = max(Populationobjs,[],1);
                fmin  = min(Populationobjs,[],1);
                PCObj = (Populationobjs-repmat(fmin,size(Population,1),1))./repmat(fmax-fmin,size(Population,1),1);
                % Determine the radius of the niche
                d  = pdist2(PCObj,PCObj);
                d(logical(eye(length(d)))) = inf;
                sd = sort(d,2);
                num_obj = size(Populationobjs,2);
                r  = median(sd(:,min(num_obj,size(sd,2))));
                R  = min(d./r,1);
                % Delete solution one by one
                while length(Population) > nPop
                    [~,worst]  = max(1-prod(R,2));
                    Population(worst,:)  = [];
                    R(worst,:) = [];
                    R(:,worst) = [];
                    W(worst,:) = [];
                end
            else
                Index = find(bi==mode(bi));
                Value_Tche2 = max(abs(Populationobjs(Index,:)-repmat(Z,size(Index,1),1))./W(Index,:),[],2);
                Index_max= find(Value_Tche2 == max(Value_Tche2));
                Population(Index(Index_max(1)),:)=[];
                W(Index(Index_max(1)),:)=[];
            end
        end
        % Update the neighbours of each weight
        B = pdist2(W,W);
        [~,B] = sort(B,2);
        B = B(:,1:T);
    end
    iter = iter + 1
end

DomiPop = DecodeFunction(Archive,XVessel,YVessel,RVessel,GenotypeLength,size(Archive,1),VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition); % 计算初始种群的目标函数值
Epa = DomiPop(:,161:163);
R = [520 260 2.5];
hypervolume(Epa, R, 100000)