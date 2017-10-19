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
filename = 'training2.csv'
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
	
class IterRegistry(type):
	"""A Metaclass that allows a class to be iterable"""
	def __iter__(cls):
		return iter(cls._registry)

class Sensor(object):
	"""Creates sensor class for IR and sonar sensors"""
	__metaclass__ = IterRegistry
	_registry = []
	
	
	def __init__(self, min_range,max_range,var,type_, data_readings, a=0, b=0,\
	rolling_var=np.empty([len(index)])):
		
		self._registry.append(self)
		self.min_ = min_range
		self.max_ = max_range
		self.var = var
		self.type_ = type_.lower()
		self.within_range = False
		self.data = data_readings
		self.a = a
		self.b = b
		self.roll_var = rolling_var
		
	def __str__(self):
		return('Type:{}\nRange:{}-{}m\nVariance:{}'.format(self.type_,self.min_,self.max_,self.var))
	
	def within_sensor_range(self,estimate):
		if estimate < self.min_ or estimate > self.max_:
			self.within_range = False
		else:
			self.within_range = True
			
	def is_ir(self):
		if self.type_ == 'ir':
			return True
		else:
			return False
			
	def ir_range(self,type_):
		if self.type_ == 'ir':
			distance = self.a/(self.data - self.b)
		else:
			distance = self.data
		return distance

def LUT(xdata,ydata,xvalue,res=100):
	i = 0
	if len(xdata) != len(ydata):
		sys.stderr.write("xdata and ydata are not same lenghth")
	# Check if xvalue is within window range
	while True:
		#~ print("X index {}".format(((i+1)*len(xdata))//res))
		#~ print "{} < {} < {}".format(xdata[(i*len(xdata))//res],xvalue,xdata[((i+1)*len(xdata))//res])
		if ((xvalue >= xdata[(i*len(xdata))//res]) and \
		(xvalue <= xdata[((i+1)*len(xdata))//res])):
			break
		
		i = i + 1 #iterate to next window of LUT	
		
		if i == 99:
			break
	#~ print ("Xval {}  Y index {}  Y length {}".format(xvalue,((i+1)*len(ydata)//res),len(ydata)))
	
	if ((i+1)*len(xdata))//res >= len(ydata):
		yvalue = ydata[len(ydata)-1]
	else:
		yvalue = ydata[((i+1)*len(xdata))//res] #result
	return yvalue

def poly2(abc_vals,x):
	"""polynomial in form ax^2+bx+c"""
	y = abc_vals[0]*x**2 + abc_vals[1]*x + abc_vals[2]
	return y
	
def get_sensor_variance(pos):
    '''
    Calculate sensor variances as a function of most recent position estimate
    '''
    # Define sensors useful ranges
    sonar2_cutoff = 0.6
    ir1_cutoff = (0.15, 0.3)
    ir2_cutoff = (0.2, 0.3)
    ir3_cutoff = (0.15, 0.5)
    ir4_cutoff = 2.5
    #~ sonar2_cutoff = 0.6
    #~ ir1_cutoff = (0.15, 0.3)
    #~ ir2_cutoff = (0.2, 0.3)
    #~ ir3_cutoff = 1.2
    #~ ir4_cutoff = 1.5

    # Coefficients for exponential variance models
    sonar1_coeff = (0.002977, 2.25556149)
    ir3_coeff = (2.1495e-5, 12.70930048)       # Better results achieved by tweaking model manually. Not sure why
    ir4_coeff =  (0.12185633 ,1.43907183)

    ir3_error_model = ir3_coeff[0] * np.exp(ir3_coeff[1] * pos)
    ir4_error_model = ir4_coeff[0] * np.exp(ir4_coeff[1] * pos)

    var_sonar1 = sonar1_coeff[0] * np.exp(sonar1_coeff[1]*pos)
    var_sonar2 = 2**64 if pos < sonar2_cutoff else 0.015
    var_ir1 = 2**64 if pos < ir1_cutoff[0] or pos > ir1_cutoff[1] else 0.002
    var_ir2 = 2**64 if pos < ir2_cutoff[0] or pos > ir2_cutoff[1] else 0.00103
    var_ir3 = 2**64 if pos < ir3_cutoff[0] or pos > ir3_cutoff[1] else ir3_error_model
    var_ir4 = 2**64 if pos < ir4_cutoff else ir4_error_model
    
    variance_vector = [var_sonar1, var_sonar2, var_ir1, var_ir2, var_ir3, var_ir4]
    return variance_vector
    
def fuse_sensors(data_vec, var_vec):
    '''
    Fuse sensor data for kalman filter
    '''
    num = denom = 0 
    for data,var in zip(data_vec,var_vec):
        num += (1 / var) * data
        denom += 1 / var
    data = num / denom
    var_v = 1 / denom
    return data, var_v
    

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
	
# Convert raw_ir voltages to distances
ir1 = ir_voltage_to_range(raw_ir1,0.1660,-0.0022)
ir2 = ir_voltage_to_range(raw_ir2,0.1560,0.0473)
ir3 = ir_voltage_to_range(raw_ir3,0.2848,0.1086)
ir4 = ir_voltage_to_range(raw_ir4,1.5724,1.2021)


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
var_post = []



# Sensor lower limits (all in meters)
sonar1_min_max = [0.0,4]
sonar2_min_max = [0.45,5]
ir1_min_max = [0.15,1]
ir2_min_max = [0.2,0.4]
ir3_min_max = [0.15,0.5]
ir4_min_max = [2.5,5]

# IR variances found in matlab with ALL of the given training data
var_IR = [0.3072, 0.2260, 0.8559, 0.8330]
var_sonar = [0.5396, 0.0336]

var_sensors = var_sonar[0]

# LSE best line of fit coefficients
a = [0.1627, 0.1558, 0.2925, 1.5664]
b = [-0.0022, 0.0549, 0.1024, 1.2081]

sonar1_obj = Sensor(sonar1_min_max[0],sonar1_min_max[1],var_sonar[0],'Sonar',sonar1)
sonar2_obj = Sensor(sonar2_min_max[0],sonar2_min_max[1],var_sonar[1],'Sonar',sonar2)
#~ ir1_obj = Sensor(ir1_min_max[0],ir1_min_max[1],var_IR[0],'ir',raw_ir1,a[0],b[0]) 
#~ ir2_obj = Sensor(ir2_min_max[0],ir2_min_max[1],var_IR[1],'ir',raw_ir2,a[1],b[1]) # Improves performance!
#~ ir3_obj = Sensor(ir3_min_max[0],ir3_min_max[1],var_IR[2],'ir',raw_ir3,a[2],b[2]) # Improves performance!
#~ ir4_obj = Sensor(ir4_min_max[0],ir4_min_max[1],var_IR[3],'ir',raw_ir4,a[3],b[3])

#Sort range in ascending order of values
sorted_range = pd.DataFrame(data=range_)
sorted_range = sorted_range.sort_values([0])
sorted_range = sorted_range[0].tolist()

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
	rollvar_sensor = pd.DataFrame(data=sensor.data,index=range_)
	rollvar_sensor = rollvar_sensor.sort_index()
	rollvar_sensor = pd.rolling_var(rollvar_sensor,window) #Calculate Var
	rollvar_sensor = rollvar_sensor[0].tolist()
	if ir3_min_max == [sensor.min_,sensor.max_]:
		rollvar_sensor = [x * 100 for x in rollvar_sensor]
	sensor.roll_var = rollvar_sensor
	datttt += 1

# Rolling variance of the process noise
range_command = [0]
for i in range(1,len(velocity_command)):
	range_command.append(range_command[i-1]+velocity_command[i]*(time[i]-time[i-1]))
	
rollvar_W = pd.DataFrame(data=range_command,index=range_)
rollvar_W = rollvar_W.sort_index()
rollvar_W = pd.rolling_std(rollvar_W,window)**2

# Load test data for kalman application
#~ filename = 'test.csv'
#~ data = np.loadtxt(filename, delimiter=',', skiprows=1)

#~ # Split into columns
#~ index, time, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

#~ # Load test data into all sensors
#~ ir1_obj.data = raw_ir1
#~ ir2_obj.data = raw_ir2
#~ ir3_obj.data = raw_ir3
#~ ir4_obj.data = raw_ir4
#~ sonar1_obj.data = sonar1
#~ sonar2_obj.data = sonar2

# Process noise variance
var_W = 9.3e-6 #Max update

# Start with a estimate of starting position
x_post = 0.0
var_X_post = 1.0


for i in range(1,len(time)):
	dt = time[i] - time[i-1]
	
	print x_post
	
	if x_post < 0:
		x_post = 0
	
	# Calculate prior estimate of position and its variance ( using motion model )
	x_prior = x_post + v[i-1] * dt
	var_X_prior = var_X_post + var_W
	
	data_vec = []
	
	data_vec = [sonar1[i], sonar2[i], ir1[i], ir2[i], ir3[i], ir4[i]]
	#~ for sensor in Sensor:
		#~ data_vec.append(sensor.ir_range(sensor.type_)[i])
	
	z_denom = 0
	z_nume = 0
	for sensor in Sensor:
		#~ print x_post
		sensor.within_sensor_range(x_post)
		print sensor
		if sensor.within_range:
			sensor_var = LUT(sorted_range,sensor.roll_var,x_post)
			if np.isnan(sensor_var):
				sensor_var = sensor.var
			#~ sensor.data = sensor.ir_range(sensor.type_)
			z_nume += sensor.data[i]/sensor_var
			z_denom += 1/sensor_var
	if z_denom == 0:
		z = z_nume
	else:
		var_sensors = 1/(z_denom)
		z = z_nume/z_denom
		
	#~ var_vec = get_sensor_variance(x_post)
	
	#~ z, var_sensors = fuse_sensors(data_vec, var_vec)
	
	# Estimate position from measurement ( using sensor model )
	x_infer = z if z > 0 else 0 # ensures sensors can't give a negative value
	
	# Calculate Kalman gain
	K = var_X_prior / ( var_sensors + var_X_prior )
	#~ print(K)
	# Caclculate posterior estimate of position and its variance
	x_post = x_prior + K * (x_infer - x_prior)
	#~ x_post = K * x_infer + (1 - K) * x_prior # Max's method (same output)
	
	var_X_post = 1/(1/var_sensors+1/var_X_prior)
	#~ var_X_post = (1 - K) * var_X_prior # Max's method (same output)
	
	x_array.append(x_post)
	t_array.append(time[i])
	z_array.append(z)
	k_array.append(K)
	range_array.append(range_[i])
	MSE_array.append(range_[i] - x_post) # gets estimate error
	var_post.append(var_X_post)
	
MSE = 0
MSE_z = 0

for i in range(len(MSE_array)):
	MSE += MSE_array[i]**2
	MSE_z += (range_array[i] - z_array[i])**2 # gets sensor fusion error squared
	
MSE_z = MSE_z/len(MSE_array)
MSE = MSE/len(MSE_array)

print("MSE for estimate:%",MSE,"MSE for sensor measurements:",MSE_z)


# Plot distance estimate
plt.figure()
plt.plot(t_array, range_array, 'g-', alpha=0.2)
plt.plot(t_array, range_command[1:], 'b-', alpha=0.2)
plt.plot(t_array, x_array, 'r.', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Distance Estimate')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
# Plot z
plt.figure()
plt.plot(t_array, range_array, 'g-', alpha=0.2)
plt.plot(t_array, range_command[1:], 'b-', alpha=0.2)
plt.plot(t_array, z_array, '.', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Measured')
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
# Plot X post variance values
#~ plt.figure()
#~ plt.plot(t_array, var_post, 'b.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Var Post X')
#~ plt.xlabel('Time (s)')
#~ plt.ylabel('Var (m)')

#~ # Plot Rolling Variances
################################# Process:
#~ plt.figure()
#~ plt.plot(sorted_range, rollvar_W, '.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Sorted Process Roll Var')
#~ plt.xlabel('Range')
#~ plt.ylabel('Variance')
################################# Sonar:
#~ plt.figure()
#~ plt.plot(sorted_range,sonar2_obj.roll_var, '.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Sorted Roll Var sonar2')
#~ plt.xlabel('Range')
#~ plt.ylabel('Variance')
#~ plt.figure()
#~ plt.plot(sorted_range,sonar1_obj.roll_var, '.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Sorted Roll Var sonar1')
#~ plt.xlabel('Range')
#~ plt.ylabel('Variance')
#~ ################################# IR:
#~ plt.figure()
#~ plt.plot(sorted_range,rollvar_ir1, '.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Sorted Roll Var IR1')
#~ plt.xlabel('Range')
#~ plt.ylabel('Variance')
#~ plt.figure()
#~ plt.plot(sorted_range, rollvar_ir2, '.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Sorted Roll Var IR2')
#~ plt.xlabel('Range')
#~ plt.ylabel('Variance')
#~ plt.figure()
#~ plt.plot(sorted_range, rollvar_ir3, '.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Sorted Roll Var IR3')
#~ plt.xlabel('Range')
#~ plt.ylabel('Variance')
#~ plt.figure()
#~ plt.plot(sorted_range, rollvar_ir4, '.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Sorted Roll Var IR4')
#~ plt.xlabel('Range')
#~ plt.ylabel('Variance')
################################
plt.show()

