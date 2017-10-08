# Kalman filter demonstration program to estimate
# position of a Dalek moving at constant speed .
# M . P . Hayes UCECE 2015
from numpy.random import randn
import matplotlib.pyplot as plt
import numpy as np


# Load data
filename = 'calibration.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

# Initial position + time
x = range_(0)
t = time(0)
# Speed
v = velocity_command
# Time step
dt = 0.6

x_array = []
t_array = []
k_array = []
z_array = []

# Position process noise standard deviation
std_W = 0.02 * dt
# Measurement noise standard deviation
std_V = 0.1
# Process noise variances
var_W = std_W ** 2

# Measurement noise variances
var_W = std_V ** 2

var_S1 = std_V ** 2
var_S2 = std_V ** 2

var_IR1 = std_V ** 2
var_IR2 = std_V ** 2
var_IR3 = std_V ** 2
var_IR4 = std_V ** 2

# Start with a poor initial estimate of Dalek 's position
x_post = 10
var_X_post = 10 ** 2

#~ limit time to 50 seconds
for i in range(len(range_)):
	# Update simulated Dalek position
	x = x + v * dt + randn (1) * var_W
	t = t + dt
		
	# Calculate prior estimate of position and its variance ( using motion model )
	x_prior = x_post + v * dt
	var_X_prior = var_X_post + var_W

	# Measure range
	z = x + randn (1) * std_V


	# Estimate position from measurement ( using sensor model )
	x_infer = z

	# Calculate Kalman gain
	K = var_X_prior / ( var_V + var_X_prior )
	# Caclculate posterior estimate of position and its variance
	x_post = x_prior + K * ( x_infer - x_prior )
	var_X_post = (1 - K ) * var_X_prior
	
	v = v + a
	
	x_array.append(x)
	t_array.append(t)
	z_array.append(z)
	k_array.append(K)

# Plot Dalek distance
plt.figure()
plt.plot(t_array, x_array, '.', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Distance Estimator')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
# Plot z
plt.figure()
plt.plot(t_array, z_array, '-', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Measured')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
# Plot k gain
plt.figure()
plt.plot(t_array, k_array, '.', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Kalman Gain')
plt.xlabel('Time (s)')
plt.ylabel('Gain')
plt.show()

