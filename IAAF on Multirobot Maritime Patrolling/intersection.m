function intersect = intersection(co2)

% plot(co2(:,1),co2(:,2),'-*')
intersect = 0;

if length(co2) > 4
    
    for i = 1:length(co2) - 3
        for j = i+2:length(co2)-2
            x = [co2(i,1), co2(i+1,1),co2(j,1), co2(j+1,1)];
            y = [co2(i,2), co2(i+1,2),co2(j,2), co2(j+1,2)];
            
            dt1=det([1,1,1;x(1),x(2),x(3);y(1),y(2),y(3)])*det([1,1,1;x(1),x(2),x(4);y(1),y(2),y(4)]);
            dt2=det([1,1,1;x(1),x(3),x(4);y(1),y(3),y(4)])*det([1,1,1;x(2),x(3),x(4);y(2),y(3),y(4)]);
            
            if(dt1<=0 & dt2<=0)
                intersect = 1;
                break
            end
        end
    end
end
% intersect
