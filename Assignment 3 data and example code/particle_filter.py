#!/usr/bin/env python3
"""Example code for ENMT482 assignment."""

import numpy as np
import matplotlib.pyplot as plt
import bisect

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
	
# Load data
filename = 'training2.csv'
data = np.loadtxt(filename, delimiter=',', skiprows=1)

# Split into columns
index, time, range_, velocity_command, raw_ir1, raw_ir2, raw_ir3, raw_ir4, sonar1, sonar2 = data.T




    

