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
	
def smart_filter(range_,min_,max_):
	"""Filter out any unexpectedly high or low range values"""
	for i in range(len(range_)):
		if range_[i] > max_ or range_[i] < min_:
				#~ range_[i] = range_[i-1]
				range_[i] = 0
	return range_

# Load data
filename = 'training2.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T



# Plot true range and sonar measurements over time
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(time, range_)
plt.xlabel('Time (s)')
plt.ylabel('Range (m)')
plt.title('True range')

plt.subplot(132)
plt.plot(time, sonar1, '.', alpha=0.2)
plt.plot(time, range_)
plt.title('Sonar1')
plt.xlabel('Time (s)')

#~ plt.subplot(133)
#~ plt.plot(time, sonar2, '.', alpha=0.2)
#~ plt.plot(time, range_)
#~ plt.title('Sonar2')
#~ plt.xlabel('Time (s)')

plt.subplot(133)
ir3_range = ir_voltage_to_range(raw_ir3,0.2848,0.1086)
plt.plot(time, smart_filter(ir3_range,0,5), '.', alpha=0.2)
plt.plot(time, range_)
plt.title('IR1')
plt.xlabel('Time (s)')

# Plot sonar error

# Generate error values for each sonar sensor
sonar1_error = range_ - sonar1
sonar2_error = range_ - sonar2

# Measurement noise variances
var_S1 = var(sonar1_error)
var_S2 = var(sonar2_error)

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.plot(time, range_ - sonar1, '.', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Sonar1 error')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')

plt.subplot(122)
plt.plot(time, range_ - sonar2, '.', alpha=0.2)
plt.axhline(0, color='k')
plt.title('Sonar2 error')
plt.xlabel('Time (s)')


# Plot IR sensor measurements
plt.figure(figsize=(8, 7))

plt.subplot(221)
plt.plot(range_, raw_ir1, '.', alpha=0.5)
plt.title('IR1')
plt.ylabel('Measurement (V)')

plt.subplot(222)
plt.plot(range_, raw_ir2, '.', alpha=0.5)
plt.title('IR2')

plt.subplot(223)
plt.plot(range_, raw_ir3, '.', alpha=0.5)
plt.title('IR3')
plt.xlabel('Range (m)')
plt.ylabel('Measurement (V)')

plt.subplot(224)
plt.plot(range_, raw_ir4, '.', alpha=0.5)
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
    

