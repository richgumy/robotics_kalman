%% Findng equations for IR sensors using least squares
% Assuming the IR equations are in the form voltage = a*distance^(-1*b)
% co-efficients a and b are found.
% Outputs average variance for each sonar sensor data
% TO DO:
%   - Create a variance function as a function of range Var_sonar(x)

% • sonar1 is an HC-SR04 sonar rangefinder (0.02–4m)
% • sonar2 is a MaxBotix HRUSB-MaxSonar-EZ MB1433 sonar rangefinder (0.3–5m)
   
function [vari] = sonar_var(filename)
    close all
    mat = csvread(filename,1,1);

    time = mat(:,1);
    range = mat(:,2);
    velocity_cmd = mat(:,3);
    sonar1 = mat(:,8);
    sonar2 = mat(:,9);
    
%     sonar_min_max = [0.02 4;
%                     0.3 5];
    sonar_min_max = [0 5;
                    0 5];
    
    vari = zeros(2,1);
    
    
    raw_sonar_all = [sonar1 sonar2];
        
    for i=1:2
        index = 1;
        filtered_range = 0;
        filtered_sonar = 0;
        filtered_time = 0;
        filtered_cmd_range = 0;
        for j=2:length(range)
            if range(j) > sonar_min_max(i,1) && range(j) < sonar_min_max(i,2)
                filtered_range(index) = range(j);
                filtered_sonar(index) = raw_sonar_all(j,i);
                filtered_time(index) = time(j);
                if index-1 > 0
                    filtered_cmd_range(index) = filtered_cmd_range(index-1) + velocity_cmd(j)*(time(j)-time(j-1));
                else
                    filtered_cmd_range(index) = velocity_cmd(j)*(time(j)-time(j-1));
                end
                index = index + 1;
            end
        end            
        
        figure
        plot(filtered_time,filtered_sonar,'ko',time,range,'b-');
        ystring = sprintf('Sonar%d Range',i);
        xlabel('Time');ylabel(ystring);
        figure
        plot(filtered_range,filtered_range - filtered_cmd_range,'c.');
        process_noise = var(filtered_range - filtered_cmd_range)
        error = filtered_range - filtered_sonar;
        vari(i) = sum((error - sum(error)/length(error)).^2)/(length(error)-1);
        var(error)
    end
end
