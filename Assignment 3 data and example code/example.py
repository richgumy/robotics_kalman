#!/usr/bin/env python3
"""Example code for ENMT482 assignment."""

import numpy as np
import matplotlib.pyplot as plt

def var(sample):
	"""Find variance of a desired data set"""
	sample_mean = sum(sample)/len(sample)
	vari = sum((sample - sample_mean)**2)/len(sample-1)
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
	
	
	def __init__(self, min_range,max_range,var,type_, data_readings, a=0, b=0):
		
		self._registry.append(self)
		self.min_ = min_range
		self.max_ = max_range
		self.var = var
		self.type_ = type_.lower()
		self.within_range = False
		self.data = data_readings
		self.a = a
		self.b = b
		
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
	
# Load data
filename = 'training2.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T

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
ir1_obj = Sensor(ir1_min_max[0],ir1_min_max[1],var_IR[0],'ir',raw_ir1,a[0],b[0]) 
ir2_obj = Sensor(ir2_min_max[0],ir2_min_max[1],var_IR[1],'ir',raw_ir2,a[1],b[1]) # Improves performance!
ir3_obj = Sensor(ir3_min_max[0],ir3_min_max[1],var_IR[2],'ir',raw_ir3,a[2],b[2]) # Improves performance!
ir4_obj = Sensor(ir4_min_max[0],ir4_min_max[1],var_IR[3],'ir',raw_ir4,a[3],b[3])


for sensor in Sensor:
	sensor.data = sensor.ir_range(sensor.type_)
	#~ for i in range(len(time)):
		#~ if not sensor.within_sensor_range(range_[i]):
			#~ sensor.data[i] = 0

# Plot true range and sonar measurements over time
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(time, range_)
plt.xlabel('Time (s)')
plt.ylabel('Range (m)')
plt.title('True range')

#~ plt.figure(figsize=(12, 5))

#~ plt.subplot(121)
#~ plt.plot(time, range_ - sonar1, '.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Sonar1 error')
#~ plt.xlabel('Time (s)')
#~ plt.ylabel('Error (m)')

#~ plt.subplot(122)
#~ plt.plot(time, range_ - sonar2, '.', alpha=0.2)
#~ plt.axhline(0, color='k')
#~ plt.title('Sonar2 error')
#~ plt.xlabel('Time (s)')


# Plot IR sensor measurements
plt.figure(figsize=(16, 10))

plt.subplot(231)
plt.plot(time, sonar1_obj.data, '.', alpha=0.2)
plt.plot(time, range_)
plt.title('Sonar1')
plt.xlabel('Time (s)')

plt.subplot(232)
plt.plot(time, sonar2_obj.data, '.', alpha=0.2)
plt.plot(time, range_)
plt.title('Sonar2')
plt.xlabel('Time (s)')

plt.subplot(233)
plt.plot(time, ir1_obj.data, '.', alpha=0.5)
plt.plot(time, range_)
plt.title('IR1')
plt.ylabel('Measurement (V)')

plt.subplot(234)
plt.plot(time, ir2_obj.data, '.', alpha=0.5)
plt.plot(time, range_)
plt.title('IR2')

plt.subplot(235)
plt.plot(time, ir3_obj.data, '.', alpha=0.5)
plt.plot(time, range_)
plt.title('IR3')
plt.xlabel('Range (m)')
plt.ylabel('Measurement (V)')

plt.subplot(236)
plt.plot(time, ir4_obj.data, '.', alpha=0.5)
plt.plot(time, range_)
plt.title('IR4')
plt.xlabel('Range (m)')
plt.show()



# You might find these helpful if you want to implement a particle filter

import bisect

def resample(particles, weights):
    """Resample particles in proportion to their weights.

    Particles and weights should be arrays, and will be updated in place."""

    cum_weights = np.cumsum(weights)
    cum_weights /= cum_weights[-1]

    new_particles = []
    for _ in particles:
        # Copy a particle into the list of new particles, choosing based
        # on weight
        m = bisect.bisect_left(cum_weights, np.random.uniform(0, 1))
        p = particles[m]
        new_particles.append(p)

    # Replace old particles with new particles
    for m, p in enumerate(new_particles):
        particles[m] = p

    # Reset weights
    weights[:] = 1


def is_degenerate(weights):
    """Return true if the particles are degenerate and need resampling."""
    w = weights/np.sum(weights)
    return 1/np.sum(w**2) < 0.5*len(w)
    

