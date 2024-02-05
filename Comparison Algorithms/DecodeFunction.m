function DecodePop = DecodeFunction(InitPop,XVessel,YVessel,RVessel,GenotypeLength,PopulationSize,VarietyNumber,MaxRobotNo,DistanceWeight,PortPosition) % �����ʼ��Ⱥ��Ŀ�꺯��ֵ
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
                    varietycode')'/(2.^GenotypeLength(k)-1)*RVessel(j); % �뾶�Ľ���
            end
            if mod(k,4) == 2
                DecodePop(i,VarietyNumber*sum(GenotypeLength)+length(GenotypeLength)*(j-1)+k)=(2.^(GenotypeLength(k)-1:-1:0)*...
                    varietycode')'/(2.^GenotypeLength(k)-1)*2*pi; % �ǶȵĽ���
            end
            if mod(k,4) == 3
                DecodePop(i,VarietyNumber*sum(GenotypeLength)+length(GenotypeLength)*(j-1)+k)=min(floor((2.^(GenotypeLength(k)-1:-1:0)*...
                    varietycode')'/(2.^GenotypeLength(k)-1)*MaxRobotNo)+1,MaxRobotNo); % ���Ľ���
                CateNumber = [CateNumber min(floor((2.^(GenotypeLength(k)-1:-1:0)*...
                    varietycode')'/(2.^GenotypeLength(k)-1)*MaxRobotNo)+1,MaxRobotNo)]; % �洢���ı��
            end
        end
    end
    f1 = 0;
    CateType = [];
    RankType = zeros(1,length(XVessel));
    EachPath = [];
    for s = 1:MaxRobotNo % ����ÿһ��
        vindex = find(CateNumber==s); % ���ڸ���Ĵ�ֻ�����
        if isempty(vindex) % �������û�д�ֻ���򲻼���
            continue
        end
        if length(vindex)>0 % ����Щ����ֵ���ż�һ��
            CateType = [CateType s];
        end
        RobotNoinCate = length(vindex); % ���ബֻ�ĸ���
        vesselvalue = []; % ���ബֻ�����Ĵ�С����
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
        for m = 1:RobotNoinCate % �����ڸ����λ�����
            DecodePop(i,VarietyNumber*sum(GenotypeLength)+length(GenotypeLength)*(vindex(m)-1)+4) = m; % �ڸ���˳��Ľ���
            RankType(vindex(m)) = m;
            polarcoor = [polarcoor;DecodePop(i,VarietyNumber*sum(GenotypeLength)+length(GenotypeLength)*(vindex(m)-1)+1) DecodePop(i,VarietyNumber*sum(GenotypeLength)+length(GenotypeLength)*(vindex(m)-1)+2)]; % �ź���ļ�����
            anglecoor(m,1) = XVessel(vindex(m))+polarcoor(m,1)*cos(polarcoor(m,2)); % �����ֱ������
            anglecoor(m,2) = YVessel(vindex(m))+polarcoor(m,1)*sin(polarcoor(m,2));
        end
        % �ü�һ���ۿڵ�λ����Ϊ��ʼ��ͽ�����
        anglecoor = [PortPosition;anglecoor;PortPosition];
        path1 = 0;
        for m = 1:RobotNoinCate+1 % �����һ��Ŀ�꺯����min����֮��
            f1 = f1+sqrt((anglecoor(m,1)-anglecoor(m+1,1))^2+(anglecoor(m,2)-anglecoor(m+1,2))^2);
            %             plot([anglecoor(m,1),anglecoor(m+1,1)],[anglecoor(m,2),anglecoor(m+1,2)])
            path1 = path1+sqrt((anglecoor(m,1)-anglecoor(m+1,1))^2+(anglecoor(m,2)-anglecoor(m+1,2))^2);
        end
        EachPath = [EachPath path1];
    end
    DecodePop(i,(sum(GenotypeLength)+length(GenotypeLength))*length(XVessel)+1) = round(f1); % �洢��һ��Ŀ�꺯������С������֮�ͣ���������
    CateType = unique(CateType);
    %     f2 = length(CateType);
    f2 = max(EachPath);
    DecodePop(i,(sum(GenotypeLength)+length(GenotypeLength))*length(XVessel)+2) = round(f2); % �洢�ڶ���Ŀ�꺯������С�����ĸ������������˵ĸ�������������
    f3 = sum(DistanceWeight.*RankType);
    DecodePop(i,(sum(GenotypeLength)+length(GenotypeLength))*length(XVessel)+3) = f3; % �洢������Ŀ�꺯������С��������ʧ
end