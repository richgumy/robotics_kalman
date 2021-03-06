%% Findng equations for IR sensors using least squares
% Assuming the IR equations are in the form voltage = a*distance^(-1*b)
% co-efficients a and b are found.
% Outputs a(1-4) and b(1-4) for each IR sensor for a given training input
% file

% � sonar1 is an HC-SR04 sonar rangefinder (0.02�4m)
% � sonar2 is a MaxBotix HRUSB-MaxSonar-EZ MB1433 sonar rangefinder (0.3�5m)
% � ir1 is a Sharp GP2Y0A02YK0F infrared rangefinder (0.15�1.5m)
% � ir2 is a Sharp GP2Y0A41SK0F infrared rangefinder (4�30cm)
% � ir3 is a Sharp GP2Y0A21YK infrared rangefinder (10�80cm)
% � ir4 is a Sharp GP2Y0A710K0F infrared rangefinder (1�5m)
   
function [e] = irLeastSquares(filename)
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
    
    % Specified Sensor ranges
    sensors_min_max_values = [0.02 3;
                    0.4 5;
                    0.15 0.3;
                    0.04 0.3;
                    0.1 0.8;
                    1 5];
    % Max range
%     sensors_min_max_values = [0 5;
%                     0 5;
%                     0 5;
%                     0 5];

    % Ranges should be limited with no upper limit as these sensors will
    % be neglected as their variance is so large
%     sensors_min_max_values = [0.3 5;
%                     0.3 5;
%                     0.55 5;
%                     1 5];
%                
    a = zeros(4,1);
    b = a;
    
    raw_sensors_all = [sonar1 sonar2 raw_ir1 raw_ir2 raw_ir3 raw_ir4];
        
    for i=1:6
        index = 1;
        filtered_range = 0;
        filtered_ir = 0;
        filtered_time = 0;
        for j=1:length(range)
            if range(j) > sensors_min_max_values(i,1) && range(j) < sensors_min_max_values(i,2)
                filtered_range(index) = range(j);
                filtered_ir(index) = raw_sensors_all(j,i);
                filtered_time(index) = time(j);
                index = index + 1;
            end
        end
        
        inv_poly_funct = @(x,xdata) x(1)*xdata.^(-1) + x(2);
        linear_eqn = @(a,xdata) a(1)*xdata + a(2);
        
        if (i > 2)
            x0 = [1 1];

    %         x = lsqcurvefit(linear_eqn,x0,filtered_range.^(-1),filtered_ir);
            x = lsqcurvefit(inv_poly_funct,x0,filtered_range,filtered_ir);

            a(i) = x(1);
            b(i) = x(2);

            filtered_ir = inv_poly_funct(x,filtered_ir);
        end

%         x_range = linspace(min(filtered_range.^(-1)),max(filtered_range.^(-1)));
        x_range = linspace(min(filtered_range),max(filtered_range),length(filtered_range));   
        
        [sorted_range, ind] = sort(filtered_range);
        sorted_ir = zeros(1,length(sorted_range));
%         new_error = zeros(size(error));
        for m=1:length(sorted_range)
%             new_error(m) = error(ind(m));
            sorted_ir(m) = filtered_ir(ind(m));
        end
        
        var_dist = movvar(sorted_ir,20);
        
        poly2_funct = @(x,xdata)  x(1)*xdata.^2 + x(2)*xdata + x(3);
        poly4_funct = @(x,xdata)  x(1)*xdata.^4 + x(2)*xdata.^2 + x(3);
        exp_funct = @(x,xdata) x(1)*exp(x(2)*xdata) + x(3);
        
        k2_0 = [0 0 0];
        k2(i,:) = lsqcurvefit(poly2_funct,k2_0,sorted_range,var_dist);
        k4_0 = [0 0 0];
        k4(i,:) = lsqcurvefit(poly4_funct,k4_0,sorted_range,var_dist);
        e_0 = [0 0 0];
        e(i,:) = lsqcurvefit(exp_funct,e_0,sorted_range,var_dist);
               
        figure
        plot(sorted_range, var_dist, 'g.',x_range,poly2_funct(k2(i,:),...
                x_range),'b-',x_range,poly4_funct(k4(i,:),x_range),'r-',...
                x_range,exp_funct(e(i,:),x_range),'k-')
        if (i > 2)
            title_ = sprintf('IR %d Variance Fit',i-2);
        else
            title_ = sprintf('Sonar %d Variance Fit',i);
        end
        title(title_)
        xlabel('Range (m)');
        ylabel('Variance (m^2)');
        
        legend('Data', 'Poly2 fit', 'Poly4 fit', 'Exp fit'); 

    end
end
