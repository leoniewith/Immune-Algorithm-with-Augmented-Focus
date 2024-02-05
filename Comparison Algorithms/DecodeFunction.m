function DecodePop = DecodeFunction(InitPop,XVessel,YVessel,RVessel,GenotypeLength,PopulationSize,VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition) % 计算初始种群的目标函数值
DecodePop = InitPop;
for i = 1:PopulationSize
    CateNumber = [];
    for j = 1:VarietyNumber
        for k = 1:length(GenotypeLength)
            if k == 1
                startpoint = 1;
                endpoint = startpoint + GenotypeLength(k) - 1;
            else
                startpoint = endpoint + 1;
                endpoint = startpoint + GenotypeLength(k) - 1;
            end
            varietycode = [];
            varietycode = DecodePop(i,(j-1)*(sum(GenotypeLength))+startpoint:(j-1)*(sum(GenotypeLength))+endpoint);
            if mod(k,4) == 1
                DecodePop(i,VarietyNumber*sum(GenotypeLength)+length(GenotypeLength)*(j-1)+k)=(2.^(GenotypeLength(k)-1:-1:0)*...
                    varietycode')'/(2.^GenotypeLength(k)-1)*RVessel(j); % 半径的解码
            end
            if mod(k,4) == 2
                DecodePop(i,VarietyNumber*sum(GenotypeLength)+length(GenotypeLength)*(j-1)+k)=(2.^(GenotypeLength(k)-1:-1:0)*...
                    varietycode')'/(2.^GenotypeLength(k)-1)*2*pi; % 角度的解码
            end
            if mod(k,4) == 3
                DecodePop(i,VarietyNumber*sum(GenotypeLength)+length(GenotypeLength)*(j-1)+k)=min(floor((2.^(GenotypeLength(k)-1:-1:0)*...
                    varietycode')'/(2.^GenotypeLength(k)-1)*MaxRobotNo)+1,MaxRobotNo); % 类别的解码
                CateNumber = [CateNumber min(floor((2.^(GenotypeLength(k)-1:-1:0)*...
                    varietycode')'/(2.^GenotypeLength(k)-1)*MaxRobotNo)+1,MaxRobotNo)]; % 存储类别的编号
            end
        end
    end
    f1 = 0;
    CateType = [];
    RankType = zeros(1,length(XVessel));
    EachPath = [];
    for s = 1:MaxRobotNo % 对于每一类
        vindex = find(CateNumber==s); % 属于该类的船只的序号
        if isempty(vindex) % 如果该类没有船只，则不计算
            continue
        end
        if length(vindex)>0 % 有哪些类出现的序号记一下
            CateType = [CateType s];
        end
        RobotNoinCate = length(vindex); % 该类船只的个数
        vesselvalue = []; % 该类船只解码后的大小矩阵
        for m = 1:RobotNoinCate
            startpoint = sum(GenotypeLength)-GenotypeLength(4)+1;
            endpoint = sum(GenotypeLength);
            varietycode = [];
            varietycode = DecodePop(i,(vindex(m)-1)*(sum(GenotypeLength))+startpoint:(vindex(m)-1)*(sum(GenotypeLength))+endpoint);
            vesselvalue = [vesselvalue (2.^(GenotypeLength(4)-1:-1:0)*varietycode')'/(2.^GenotypeLength(4)-1)];
        end
        [~,bindex] = sort(vesselvalue);
        vindex = vindex(bindex);
        polarcoor = [];
        anglecoor = [];
        for m = 1:RobotNoinCate % 解码在该类的位置序号
            DecodePop(i,VarietyNumber*sum(GenotypeLength)+length(GenotypeLength)*(vindex(m)-1)+4) = m; % 在该类顺序的解码
            RankType(vindex(m)) = m;
            polarcoor = [polarcoor;DecodePop(i,VarietyNumber*sum(GenotypeLength)+length(GenotypeLength)*(vindex(m)-1)+1) DecodePop(i,VarietyNumber*sum(GenotypeLength)+length(GenotypeLength)*(vindex(m)-1)+2)]; % 排好序的极坐标
            anglecoor(m,1) = XVessel(vindex(m))+polarcoor(m,1)*cos(polarcoor(m,2)); % 换算成直角坐标
            anglecoor(m,2) = YVessel(vindex(m))+polarcoor(m,1)*sin(polarcoor(m,2));
        end
        % 得加一个港口的位置作为起始点和结束点
        anglecoor = [PortPosition;anglecoor;PortPosition];
        path1 = 0;
        for m = 1:RobotNoinCate+1 % 计算第一个目标函数：min距离之和
            f1 = f1+sqrt((anglecoor(m,1)-anglecoor(m+1,1))^2+(anglecoor(m,2)-anglecoor(m+1,2))^2);
            %             plot([anglecoor(m,1),anglecoor(m+1,1)],[anglecoor(m,2),anglecoor(m+1,2)])
            path1 = path1+sqrt((anglecoor(m,1)-anglecoor(m+1,1))^2+(anglecoor(m,2)-anglecoor(m+1,2))^2);
        end
        EachPath = [EachPath path1];
    end
    DecodePop(i,(sum(GenotypeLength)+length(GenotypeLength))*length(XVessel)+1) = round(f1); % 存储第一个目标函数，最小化距离之和，四舍五入
    CateType = unique(CateType);
    %     f2 = length(CateType);
    f2 = max(EachPath);
    DecodePop(i,(sum(GenotypeLength)+length(GenotypeLength))*length(XVessel)+2) = round(f2); % 存储第二个目标函数，最小化类别的个数，即机器人的个数，四舍五入
    f3 = sum(DistanceWeight.*RankType);
    DecodePop(i,(sum(GenotypeLength)+length(GenotypeLength))*length(XVessel)+3) = f3; % 存储第三个目标函数，最小化收益损失
end