#!/usr/bin/env python3
"""
ENMT482 - University of Canterbury
Richie Ellingham

Kalman filter demonstration program to estimate position of a 1D robot using
6 sensors

TO DO NEXT:
	- Clean it up
	- Ensure 3D capabilities -> use matrices and vectors
	- Determine variances for x, y, and theta
	- Determine sensor Co-Variance matrix
	
"""

from numpy.random import randn
from numpy.linalg import inv
from numpy import dot
import matplotlib.pyplot as plt
import numpy as np

def get_MSE(error_array):
	MSE = 0
	for i in range(len(error_array)):
		MSE += MSE_array[i]**2
		
	MSE = MSE/len(MSE_array)
	
	return (MSE)

def var(samples):
	"""Find variance of a desired data set"""
	sample_mean = sum(samples)/len(samples)
	vari = sum((samples - sample_mean)**2)/len(samples-1)
	return vari
	
# Load test data for kalman application
filename = 'data.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype='str')

# Split into columns
time, velocity_command, rotation_command, map_x, map_y, map_theta, odom_x, odom_y, odom_theta, beacon_id, beacon_x, beacon_y, beacon_theta = data.T

print(np.sin(np.pi/2),sin_t(np.pi/2))

# Time interval just used for process noise for now
#~ dt = 0.1

#~ # Speed commanded by turtlebot
#~ v = velocity_command

#~ # Process noise variances
#~ var_W = 0.0002

#~ # IR variances found in matlab with ALL of the given training data
#~ var_beacon = [0.3072, 0.2260, 0.8559]
#~ var_odom = [0.5396, 0.0336, 0.0234]

#~ var_sensors = var_sonar[0]

#~ np.matrix# Start with a estimate of starting position
#~ x_post = np.matrix("0 0")
#~ var_X_post = 0.1

#~ """ 
#~ System equation:	x(-)(n) = A*x(+)(n-1) + B*u(n-1)
 #~ [x(n),		[x(n-1),		[(-v/w)*cos(theta(n-1)) + (v/w)*sin(theta(n-1) + w*dt),
  #~ y(n),	= 	 y(n-1),	+	(v/w)*cos(theta(n-1)) - (v/w)*cos(theta(n-1) + w*dt),
 #~ theta(n]  	theta(n-1)]							w * dt							]
 #~ Sensor equation:	z(n) = C*x(n) + V(n)
 #~ (assuming sensor measure, z, isn't a function of control input, u)
#~ """
#~ A = 1
#~ B = np.matrix("")


#~ for i in range(1,len(time)):
	#~ dt = time[i] - time[i-1]
	
	#~ ## Predict Step ##
	#~ # Calculate prior estimate of position and its variance ( using motion model )
	#~ X_prior = dot(A,X_post) + dot(B,U)
	#~ var_X_prior = dot(A, dot(var_X_post,np.transpose(A))) + var_W
	#~ ## Predict Step ##
	
	#~ ## Sensor Fusion ##
	#~ z_nume = 0
	#~ z_denom = 0
	#~ z = 0	
	#~ # Measure range
	#~ for sensor in Sensor:
		#~ sensor.within_sensor_range(x_post,sensor.min_,sensor.max_,sensor.within_range)
		#~ if sensor.within_range:
			#~ sensor.data = sensor.ir_range(sensor.type_,sensor.data)
			#~ z_nume += sensor.data[i]/sensor.var
			#~ z_denom += 1/sensor.var
	#~ if z_denom == 0:
		#~ z = z_nume
	#~ else:
		#~ var_sensors = 1/(z_denom)
		#~ covar_sensors = var_sensors * np.transpose(var_sensors)
		#~ z = z_nume/z_denom
	#~ # Estimate position from measurement ( using sensor model )
	#~ x_infer = z #- (v[i-1] * dt)
	#~ ## Sensor Fusion ##
	
	#~ ## Update Step ##
	#~ # Calculate Kalman gain
	#~ C_Var_prior_C_T = dot(C,dot(var_X_prior,np.transpose(C)))
	#~ K = dot(var_X_prior,dot(np.transpose(C),inv(C_Var_prior_C_T + covar_sensors)
	#~ # Calculate posterior estimate of position and its variance
	#~ x_post = K * x_infer + (1 - K) * x_prior
	#~ var_X_post = 1/(1/var_sensors+1/var_X_prior)
	#~ ## Update Step ##
	
	#~ # Update plot arrays
	#~ x_array.append(x_post)
	#~ t_array.append(time[i])
	#~ z_array.append(z)
	#~ k_array.append(K)
	#~ range_array.append(range_[i])
	#~ MSE_array.append(range_[i] - x_post) # gets estimate error

#~ x_MSE = get_MSE(MSE_array)
#~ print x_MSE


#~ # Plot x
#~ plt.figure()
#~ plt.plot(time, map_x, '-', alpha=0.2)
#~ plt.plot(time, odom_x, 'r.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Odom x Measuremnets')
#~ plt.xlabel('Time (s)')
#~ plt.ylabel('x Distance (m)')
#~ # Plot y
#~ plt.figure()
#~ plt.plot(time, map_y, '-', alpha=0.2)
#~ plt.plot(time, odom_y, 'r.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Odom y Measuremnets')
#~ plt.xlabel('Time (s)')
#~ plt.ylabel('y Distance (m)')
#~ plt.show()

