%% Findng equations for IR sensors using least squares
% Assuming the IR equations are in the form voltage = a*distance^(-1*b)
% co-efficients a and b are found.
% Outputs average variance for each sonar sensor data
% TO DO:
%   - Create a variance function as a function of range Var_sonar(x)

  
function [vari] = sensor_var(filename)
    close all
    mat = csvread(filename,1,1);

%     time_ns = mat(:,1);
    vel_command = mat(:,1);
    rot_command = mat(:,2);
    map_x = mat(:,3);
    map_y = mat(:,4);
    map_theta = mat(:,5);
    odom_x = mat(:,6);
    odom_y = mat(:,7);
    odom_theta = mat(:,8);
    beacon_id = mat(:,9);
    beacon_x = mat(:,10);
    beacon_y = mat(:,11);
    beacon_theta = mat(:,12);
    time = 1:length(vel_command);
     
    vari = zeros(2,1);
    
    xytheta = ['x', 'y', 'theta'];
    
    map_data = [map_x map_y map_theta];
    odom_data = [odom_x odom_y odom_theta];
        
    for i=1:3     
              
        error = map_data(:,i) - odom_data(:,i);
        if i == 3 
            angle_rollover = 0;
            roll = 0;
            for j = 1:length(error)
                if j == (length(error)+1)/2
                    roll = 0;
                end
                if error(j) > 5.8 && ~roll
                    angle_rollover = 2*pi;
                    roll = ~roll;
                end
                error(j) = error(j) - angle_rollover;
                odom_data(j,i) = odom_data(j,i) - angle_rollover;
                map_data(j,i) = map_data(j,i) - angle_rollover;
            end
        end
        
        figure
        plot(time,map_data(:,i),'ko',time,odom_data(:,i),'b-');
        ystring = sprintf('Odom%s',xytheta(i));
        xlabel('Time');ylabel(ystring)
        figure
        plot(time,error,'r.');
        ystring = sprintf('Error%s',xytheta(i));
        xlabel('Time');ylabel(ystring)       
        vari(i) = var(error);
    end
end
