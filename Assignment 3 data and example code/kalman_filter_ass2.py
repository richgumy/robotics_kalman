"""
ENMT482 - University of Canterbury
Richie Ellingham

Kalman filter demonstration program to estimate position of a 1D robot using
6 sensors

TO DO NEXT:
	- Make Classes for fucking everything OOP OOP
	- Clean it up
	- Extended Kalman?
	
"""

from numpy.random import randn
import matplotlib.pyplot as plt
import numpy as np

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
	
class IterRegistry(type):
    def __iter__(cls):
        return iter(cls._registry)

class Sensor(object):
	"""Creates sensor class for IR and sonar sensors"""
	__metaclass__ = IterRegistry
	_registry = []
	
	def __init__(self, min_range,max_range,var,type_, data_readings, a=0, b=0):
		self._registry.append(self)
		self.min_ = min_range
		self.max_ = max_range
		self.var = var
		self.type_ = type_
		self.within_range = False
		self.data = data_readings
		self.a = a
		self.b = b
		
	def __str__(self):
		return('Type:{}\nRange:{}-{}m\nVariance:{}\n'.format(self.type_,self.min_,self.max_,self.var))
	
	def within_sensor_range(self,estimate,min_,max_,within_range):
		if estimate < min_ or estimate > max_:
			self.within_range = False
		else:
			self.within_range = True
			
	def ir_range(self,type_,data):
		if type_.lower() == 'ir':
			distance = self.a/(self.data - self.b)
		else:
			distance = self.data
		return distance
	
# Load training data to get sensor variances
filename = 'training1.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

# Generate error values for each sensor
sonar1_error = range_ - sonar1
sonar2_error = range_ - sonar2
ir1_error = range_ - 

# Measurement noise variances (training1, training2)
var_S1 = var(sonar1_error) # (0.19505585259364139, 0.58587920724762343)
var_S2 = var(sonar2_error) # (1.4855465481829488, 0.90549771246436972)

Var_S1_S2 = 1/(1/var_S1+1/var_S2)

# Load test data
filename = 'test.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

# Convert raw_ir voltages to distances
ir1 = ir_voltage_to_range(raw_ir1,0.1660,-0.0022)
ir2 = ir_voltage_to_range(raw_ir2,0.1560,0.0473)
ir3 = ir_voltage_to_range(raw_ir3,0.2848,0.1086)
ir4 = ir_voltage_to_range(raw_ir4,1.5724,1.2021)

# Sensor lower limits (all in meters)
sonar1_min_max = [0.02,4]
sonar2_min_max = [0.3,5]
ir1_min_max = [0.15,1.5]
ir2_min_max = [0.04,0.3]
ir3_min_max = [0.1,0.8]
ir4_min_max = [1.0,5]

# Time interval just used for process noise for now
dt = 0.1

# Speed commanded by turtlebot
v = velocity_command

# Declare plot arrays
x_array = []
t_array = []
k_array = []
z_array = []
range_array = []

# Position process noise standard deviation
std_W = 0.02 * dt

# Process noise variances
var_W = std_W ** 2

# IR variances found in matlab with training data.. try again
var_IR = [0.3409, 0.2631, 0.6688, 0.6947]

# LSE best line of fit coefficients
a = [0.1660,0.1560,0.2848,1.5724]
b = [-0.0022,0.0473,0.1086,1.2021]

sonar1_obj = Sensor(sonar1_min_max[0],sonar1_min_max[1],var_S1,'Sonar',sonar1)
sonar2_obj = Sensor(sonar2_min_max[0],sonar2_min_max[1],var_S2,'Sonar',sonar2)
#~ ir1_obj = Sensor(ir1_min_max[0],ir1_min_max[1],var_IR[0],'ir',raw_ir1,a[0],b[0])
#~ ir2_obj = Sensor(ir2_min_max[0],ir2_min_max[1],var_IR[1],'ir',raw_ir2,a[1],b[1])
#~ ir3_obj = Sensor(ir3_min_max[0],ir3_min_max[1],var_IR[2],'ir',raw_ir3,a[2],b[2])
#~ ir4_obj = Sensor(ir4_min_max[0],ir4_min_max[1],var_IR[3],'ir',raw_ir4,a[3],b[3])

# Start with a estimate of starting position
x_post = 0.1
var_X_post = 0.1

for i in range(1,len(time)):
	dt = time[i] - time[i-1]
	
	# Calculate prior estimate of position and its variance ( using motion model )
	x_prior = x_post + v[i-1] * dt
	var_X_prior = var_X_post + var_W
	
	# Declare measurement + measurement variables
	nume = 0
	denom = 0
	z = 0
		
	# Measure range
	for sensor in Sensor:
		sensor.within_sensor_range(x_post,sensor.min_,sensor.max_,sensor.within_range)
		if sensor.within_range:
			sensor.data = sensor.ir_range(sensor.type_,sensor.data)
			nume += sensor.data[i]/sensor.var
			denom += 1/sensor.var
	if denom == 0:
		z = nume
	else:
		z = nume/denom
	
	# Estimate position from measurement ( using sensor model )
	x_infer = z - (v[i-1] * dt)
	
	# Calculate Kalman gain
	K = var_X_prior / ( Var_S1_S2 + var_X_prior )
	
	# Caclculate posterior estimate of position and its variance
	x_post = K * x_infer + (1 - K) * x_prior
	
	var_X_post = 1/(1/Var_S1_S2+1/var_X_prior)
	
	x_array.append(x_post)
	t_array.append(time[i])
	z_array.append(z)
	k_array.append(K)
	range_array.append(range_[i])

# Plot distance
plt.figure()
plt.plot(t_array, range_array, '-', alpha=0.2)
plt.plot(t_array, x_array, '.', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Distance Estimator')
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
# Plot actual x
#~ plt.figure()
#~ plt.plot(t_array, range_array, '-', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Actual')
#~ plt.xlabel('Time (s)')
#~ plt.ylabel('Distance (m)')
# Plot k gain
plt.figure()
plt.plot(t_array, k_array, '.', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Kalman Gain')
plt.xlabel('Time (s)')
plt.ylabel('Gain')
plt.show()

