#!/usr/bin/env python3
"""
ENMT482 - University of Canterbury
Richie Ellingham

Kalman filter demonstration program to estimate position of a 1D robot using
6 sensors

TO DO NEXT:
	- Make Classes for everything OOP OOP
	- Clean it up
	- Extended Kalman?
	
"""

from numpy.random import randn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

# Load training data for kalman application
filename = 'training1.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T
#~ index, time, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

def var(samples):
	"""Find variance of a desired data set"""
	sample_mean = sum(samples)/len(samples)
	vari = sum((samples - sample_mean)**2)/len(samples-1)
	return vari
	
def ir_voltage_to_range(voltage,a,b):
	"""This function uses derived coefficients a and b to map the inverse
	relationship between voltage and actual sensed range"""
	distance = a/(voltage - b)
	return distance

def get_ir_linearised(curr_V, prev_V, coeffs):
    '''
    Linearise infrared sensor function about previous datapoint
    '''
    output_array = []
    for V, prev, coeff in zip(curr_V, prev_V, coeffs):
        a, b, c = coeff
        linearised = (b/(prev - a) - c) - b/(prev - a)**2 * (V - prev)
        nonlin = b / (V - a) - c
        output_array.append(linearised)
    return output_array

def load_IR_sensor_models():
    '''
    Returns the coefficients that define the infrared sensor models
    '''
    coeffs = ((0.10171772, 0.0895424, -0.07046522),
              (-0.27042247, 0.27470697, 0.04806651),
              (0.24727172, 0.22461734, -0.01960771),
              (1.21541481, 1.54949467, -0.00284672))
    return coeffs
	
class IterRegistry(type):
	"""A Metaclass that allows a class to be iterable"""
	def __iter__(cls):
		return iter(cls._registry)

class Sensor(object):
	"""Creates sensor class for IR and sonar sensors"""
	__metaclass__ = IterRegistry
	_registry = []
	
	
	def __init__(self, min_range,max_range,var,type_, data_readings, a=0, b=0, var_coeffs=[0,0,0]):
		
		self._registry.append(self)
		self.min_ = min_range
		self.max_ = max_range
		self.var = var
		self.type_ = type_
		self.within_range = False
		self.data = data_readings
		self.a = a
		self.b = b
		self.var_coeffs = var_coeffs
		
	def __str__(self):
		return('Type:{}\nRange:{}-{}m\nVariance:{}'.format(self.type_,self.min_,self.max_,self.var))
	
	def within_sensor_range(self,estimate,min_,max_,within_range):
		if estimate < min_ or estimate > max_:
			self.within_range = False
		else:
			self.within_range = True
			
	def ir_range(self,type_):
		if type_.lower() == 'ir':
			if type(self.data) == list:
				distance = [self.a/(x - self.b) for x in self.data]
			else:
				distance = self.a/(self.data - self.b)
			
		else:
			distance = self.data
		return distance
		
	def poly2(self,x):
		"""polynomial in form ax^2+bx+c"""
		#~ print self.var_coeffs[2]
		y = self.var_coeffs[0]*x**2 + self.var_coeffs[1]*x + self.var_coeffs[2]
		if self.type_ == 'sonar': # Sonar has exponential fit
			y = self.var_coeffs[0]* np.exp(self.var_coeffs[1]*x) + self.var_coeffs[2]
		return y

# Time interval just used for process noise for now
dt = 0.1

# Speed commanded by turtlebot
v = velocity_command

# Declare plot arrays
x_array = []
t_array = []
k_array = []
z_array = []
s2_array = []
range_array = []
MSE_array = []


# Process noise variance
var_W = 0.00001

# Sensor limits (all in meters)
sonar1_min_max = [0.02,4]
sonar2_min_max = [0.45,5]
ir1_min_max = [0.3,1]
ir2_min_max = [0.2,0.3]
ir3_min_max = [0.15,0.8]
ir4_min_max = [1,5]

# IR variances found in matlab with ALL of the given training data
var_IR = [0.3072, 0.2260, 0.8559, 0.8330]
var_sonar = [0.5396, 0.0336]

var_sensors = var_sonar[0]

# LSE best line of fit coefficients for IR voltage to range conversion
a = [0.1627, 0.1558, 0.2925, 1.5664]
b = [-0.0022, 0.0549, 0.1024, 1.2081]

# LSE of variance as 2nd order polynomial
var_poly_coeffs = [[0.8815, -0.3157, 0.0294],[0.2572, -0.0646, 0.0041],\
			[0.0175, -0.0067, 0.0007],[0.0014, -0.0020, 0.0011],[0.0003,1.1306,0.005]]

sonar1_obj = Sensor(sonar1_min_max[0],sonar1_min_max[1],var_sonar[0],'Sonar',sonar1,0,0,var_poly_coeffs[4]) #temporary constant var values for var_coeffs
sonar2_obj = Sensor(sonar2_min_max[0],sonar2_min_max[1],var_sonar[1],'Sonar',sonar2,0,0,[0, 0, 0.005]) # <----------'
ir1_obj = Sensor(ir1_min_max[0],ir1_min_max[1],var_IR[0],'ir',raw_ir1,a[0],b[0],var_poly_coeffs[0])
ir2_obj = Sensor(ir2_min_max[0],ir2_min_max[1],var_IR[1],'ir',raw_ir2,a[1],b[1],var_poly_coeffs[1])
ir3_obj = Sensor(ir3_min_max[0],ir3_min_max[1],var_IR[2],'ir',raw_ir3,a[2],b[2],var_poly_coeffs[2])
ir4_obj = Sensor(ir4_min_max[0],ir4_min_max[1],var_IR[3],'ir',raw_ir4,a[3],b[3],var_poly_coeffs[3])

ir1 = []
ir2 = []
ir3 = []
ir4 = []
for i in range(len(time)):
	ir_coeffs = load_IR_sensor_models()
	ir_voltages = (raw_ir1[i], raw_ir2[i], raw_ir3[i], raw_ir4[i])
	prev_V = (raw_ir1[i-1], raw_ir2[i-1], raw_ir3[i-1], raw_ir4[i-1])
	ir1_val, ir2_val, ir3_val, ir4_val = get_ir_linearised(ir_voltages, prev_V, ir_coeffs)
	ir1.append(ir1_val)
	ir2.append(ir2_val)
	ir3.append(ir3_val)
	ir4.append(ir4_val)

data_vec = [sonar1,sonar2, ir1, ir2, ir3, ir4]
datttt = 0
for sensor in Sensor:
	#Sort sensor Data based on range with window size set
	window = 100
	sensor.data = data_vec[datttt]
	datttt += 1

# Start with a estimate of starting position
x_post = 0.1
var_X_post = 0.5

for i in range(1,len(time)):
	dt = time[i] - time[i-1]
	
	# Calculate prior estimate of position and its variance ( using motion model )
	x_prior = x_post + v[i-1] * dt
	var_X_prior = var_X_post + var_W
	
	# Declare measurement + measurement variables
	z_nume = 0
	z_denom = 0
	z = 0
		
	# Measure range
	for sensor in Sensor:
		#~ print x_post
		sensor.within_sensor_range(x_post,sensor.min_,sensor.max_,sensor.within_range)
		if sensor.within_range:
			sensor.data = sensor.ir_range(sensor.type_)
			z_nume += sensor.data[i]/sensor.var
			z_denom += 1/sensor.var
	if z_denom == 0:
		z = z_nume
	else:
		var_sensors = 1/(z_denom)
		z = z_nume/z_denom
	
	# Estimate position from measurement ( using sensor model )
	x_infer = z 
	
	# Calculate Kalman gain
	K = var_X_prior / ( var_sensors + var_X_prior )
	# Caclculate posterior estimate of position and its variance
	x_post = K * x_infer + (1 - K) * x_prior
	
	var_X_post = 1/(1/var_sensors+1/var_X_prior)
	
	x_array.append(x_post)
	t_array.append(time[i])
	z_array.append(z)
	k_array.append(K)
	range_array.append(range_[i])
	MSE_array.append(range_[i] - x_post) # gets estimate error
	
	#~ if not(i % 1000):
		#~ while True:
			#~ step = raw_input("Step?")
			#~ if step == '':
				#~ break
			
		# Plot distance estimate
		#~ plt.figure()
		#~ plt.plot(t_array, range_array, '-', alpha=0.2)
		#~ plt.plot(t_array, x_array, 'r.', alpha=0.2)
		#~ plt.axhline(0, color='k')
		#~ plt.title('Distance Estimate')
		#~ plt.xlabel('Time (s)')
		#~ plt.ylabel('Distance (m)')
		#~ plt.show()
	
MSE = 0
MSE_z = 0

for i in range(len(MSE_array)):
	MSE += MSE_array[i]**2
	MSE_z += (range_array[i] - z_array[i])**2 # gets sensor fusion error squared
	
MSE_z = MSE_z/len(MSE_array)
MSE = MSE/len(MSE_array)

print(MSE,MSE_z)

#~ # Plot distance estimate
plt.figure()
plt.plot(t_array, range_array, '-', alpha=0.2)
plt.plot(t_array, x_array, 'r.', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Distance Estimate')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
# Plot z
plt.figure()
plt.plot(t_array, range_array, '-', alpha=0.2)
plt.plot(t_array, z_array, '.', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Measured')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
# Plot Sonar values
#~ plt.figure()
#~ plt.plot(time, sonar2, 'b.', alpha=0.2)
#~ plt.plot(time, sonar1, 'r.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Actual')
#~ plt.xlabel('Time (s)')
#~ plt.ylabel('Distance (m)')
plt.show()

