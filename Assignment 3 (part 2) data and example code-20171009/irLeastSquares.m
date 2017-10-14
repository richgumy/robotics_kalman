%% Findng equations for IR sensors using least squares
% Assuming the IR equations are in the form voltage = a*distance^(-1*b)
% co-efficients a and b are found.
% Outputs a(1-4) and b(1-4) for each IR sensor for a given training input
% file

% • sonar1 is an HC-SR04 sonar rangefinder (0.02–4m)
% • sonar2 is a MaxBotix HRUSB-MaxSonar-EZ MB1433 sonar rangefinder (0.3–5m)
% • ir1 is a Sharp GP2Y0A02YK0F infrared rangefinder (0.15–1.5m)
% • ir2 is a Sharp GP2Y0A41SK0F infrared rangefinder (4–30cm)
% • ir3 is a Sharp GP2Y0A21YK infrared rangefinder (10–80cm)
% • ir4 is a Sharp GP2Y0A710K0F infrared rangefinder (1–5m)
   
function [a,b,var] = irLeastSquares(filename)
    close all
    mat = csvread(filename,1,1);

    time = mat(:,1);
    range = mat(:,2);
    velocity_command = mat(:,3);
    raw_ir1 = mat(:,4);
    raw_ir2 = mat(:,5);
    raw_ir3 = mat(:,6);
    raw_ir4 = mat(:,7);
    sonar1 = mat(:,8);
    sonar2 = mat(:,9);
    
    ir_min_max_values = [0.15 1.25;
                    0.04 0.3;
                    0.1 0.8;
                    1 5];
               
    a = zeros(4,1)
    b = a;
    
    raw_ir_all = [raw_ir1 raw_ir2 raw_ir3 raw_ir4];
        
    for i=1:4
        index = 1;
        filtered_range = 0;
        filtered_ir = 0;
        filtered_time = 0;
        for j=1:length(range)
            if range(j) > ir_min_max_values(i,1) && range(j) < ir_min_max_values(i,2)
                filtered_range(index) = range(j);
                filtered_ir(index) = raw_ir_all(j,i);
                filtered_time(index) = time(j);
                index = index + 1;
            end
        end
        
        inv_poly_funct = @(x,xdata) x(1)*xdata.^(-1) + x(2);
        linearise_inv = @(a,xdata) a(1)*xdata + a(2);
        
        x0 = [1 1];
        
        x = lsqcurvefit(linearise_inv,x0,filtered_range.^(-1),filtered_ir);

        x_range = linspace(min(filtered_range.^(-1)),max(filtered_range.^(-1)));

%         figure
%         plot(filtered_range.^(-1),filtered_ir,'ko',x_range,linearise_inv(x,x_range),'b-');
%         ystring = sprintf('IR%d voltage',i);
%         xlabel('Inv range');ylabel(ystring);

        a(i) = x(1);
        b(i) = x(2);
        
        figure
        % Plot range found from LSE
        filtered_dist = a(i)./(filtered_ir-b(i));
        dist = a(i)./(raw_ir_all(:,i)-b(i));
        plot(filtered_time,filtered_dist,'ko',time,range,'b-');
        ystring = sprintf('IR%d Range',i);
        xlabel('Time');ylabel(ystring)
        
        error = filtered_range - filtered_ir;
        var(i) = sum((error - sum(error)/length(error)).^2)/(length(error)-1);
    end
end
