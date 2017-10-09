from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import bisect


###############################################################################
# Particle filter functions from part 1

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


###############################################################################
# Functions for working with 2D poses represented as (t_x, t_y, theta)

def tf_add(a, b):
    """Add two 2D transformations.
    
    Each pose is a translation followed by a rotation; adding the poses means
    doing translation A, then rotation A, then translation B (with the rotated
    axes), then rotation B.
    
    The arguments can be array-likes of broadcastable shapes, and the result
    will be the shape of whichever is larger. Here are some examples of valid
    shape combinations.
    
    Adding two transformations:
    >>> tf_add((1, 1, np.pi/2), (1, 0, 0))
    array([ 1.        ,  2.        ,  1.57079633])
    
    Adding a transformation to a list of transformations (e.g. adding predicted
    movement to particles)
    >>> tf_add([(0, 0, 0), (1, 1, np.pi/4), (1, 1, np.pi/2)], (1, 0, 0))
    array([[ 1.        ,  0.        ,  0.        ],
           [ 1.70710678,  1.70710678,  0.78539816],
           [ 1.        ,  2.        ,  1.57079633]])
    
    Adding a list of transformations to a list of transformations (e.g. adding
    noise to particles)
    >>> tf_add([(0, 0, 0), (1, 1, np.pi/4), (1, 1, np.pi/2)], [(1, 0, 0), (2, 0, 0), (3, 0, 0)])
    array([[ 1.        ,  0.        ,  0.        ],
           [ 2.41421356,  2.41421356,  0.78539816],
           [ 1.        ,  4.        ,  1.57079633]])
    
    Adding a list of transformations to a single transformation (e.g. rotating
    a list of poses to plot them better)
    >>> np.set_printoptions(suppress=True)
    >>> tf_add([(0, 0, np.pi/2)], [(1, 0, 0), (2, 0, 0), (3, 0, 0)])
    array([[ 0.        ,  1.        ,  1.57079633],
           [ 0.        ,  2.        ,  1.57079633],
           [ 0.        ,  3.        ,  1.57079633]])
    """

    # Make variables the rights shapes and types
    a = np.array(a, dtype=np.double)
    b = np.array(b, dtype=np.double)
    a_ = a.reshape(-1, 3)
    b_ = b.reshape(-1, 3)
    result = np.zeros(np.broadcast(a, b).shape, dtype=np.double).reshape((-1, 3))

    # Do the actual calculation
    result[:,0] = a_[:,0] + b_[:,0]*np.cos(a_[:,2]) - b_[:,1]*np.sin(a_[:,2])
    result[:,1] = a_[:,1] + b_[:,0]*np.sin(a_[:,2]) + b_[:,1]*np.cos(a_[:,2])
    result[:,2] = a_[:,2] + b_[:,2]

    # Make sure angle is in [0, 2pi)
    result[:,2] = (result[:,2] + 2*np.pi) % (2*np.pi)
    return result.reshape(np.broadcast(a, b).shape)


def tf_inverse(tf):
    """Return the transformation from `tf` to the origin.
    
    The argument can be a single transformation, or a list of transformations."""

    # Make variables the rights shapes and types
    tf = np.array(tf, dtype=np.double)
    tf_ = tf.reshape(-1, 3)
    result = np.zeros_like(tf_)

    # Do the actual calculation
    result[:,0] =  np.cos(tf_[:,2])*(-tf_[:,0]) + np.sin(tf_[:,2])*(-tf_[:,1])
    result[:,1] = -np.sin(tf_[:,2])*(-tf_[:,0]) + np.cos(tf_[:,2])*(-tf_[:,1])
    result[:,2] = -tf_[:,2]

    # Make sure angle is in [0, 2pi)
    result[:,2] = (result[:,2] + 2*np.pi) % (2*np.pi)
    return result.reshape(tf.shape)


def tf_between(a, b):
    """Return the transformation from `a` to `b`.
    
    The arguments can be array-likes of broadcastable shapes as in tf_add."""

    return tf_add(tf_inverse(a), b)


def range_bearing(obs):
    """Convert a fiducial observation from (x, y, theta) to (range, bearing).
    
    Range is distance from origin to point. Bearing is angle from x-axis to point.
    
    The argument can be a single obvservation, or a list of observations."""

    obs = np.array(obs, dtype=np.double).reshape(-1, 3)
    range_ = np.sqrt(obs[:,0]**2 + obs[:,1]**2)
    bearing = np.arctan2(obs[:,1], obs[:,0])
    return range_, bearing


def rot90(tfs):
    """Rotate a transformation or list of transformations by 90 degrees, so they plot better."""
    return tf_add((0, 0, np.pi/2), tfs)


###############################################################################
# Load data

# data is a (many x 13) matrix. Its columns are:
# time_ns, velocity_command, rotation_command, map_x, map_y, map_theta, odom_x, odom_y, odom_theta,
# beacon_id, beacon_x, beacon_y, beacon_theta
data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)

# Time in ns
t = data[:,0]

# Velocity command in m/s, rotation command in rad/s
command = data[:,1:3]

# Position in map frame, from SLAM
map_pos = data[:,3:6]

# Position in odometry frame, from wheel encoders and gyro
odom_pos = data[:,6:9]

# Beacon id and position in camera frame
beacon_id = data[:,9]
beacon_pos = data[:,10:]


# map_data is a (many x 13) matrix. Its columns are:
# beacon_id, x, y, theta, (9 columns of covariance)
map_data = np.genfromtxt('beacon_map.csv', delimiter=',', skip_header=1)

# Mapping from beacon id to beacon position
beacon_pos = {id_: (x, y, theta) for (id_, x, y, theta) in map_data[:,:4]}


###############################################################################
# Plots

plt.figure(figsize=(10, 5))

# Rotate beacons 90 degrees and plot
beacon_pos_rotated = rot90(map_data[:,1:4])
plt.plot(beacon_pos_rotated[:,0], beacon_pos_rotated[:,1], 'ro')

# Null out jumps in SLAM position
map_pos_clean = np.copy(map_pos)
last_good = 0
for i in range(1, len(map_pos_clean)):
    dist = np.sqrt(np.sum((map_pos_clean[i,:2] - map_pos_clean[last_good,:2])**2))
    if dist > 2:
        map_pos_clean[i] = np.nan
    else:
        last_good = i
        
# Rotate SLAM positions 90 degrees and plot
map_pos_rotated = rot90(map_pos_clean)
plt.plot(map_pos_rotated[:,0], map_pos_rotated[:,1], 'g-')

# Transform odometry positions into map frame, rotate 90 degrees and plot
odom_to_map = tf_between(odom_pos[0], map_pos[0])
odom_pos_rotated = rot90(tf_add(odom_to_map, odom_pos))
plt.plot(odom_pos_rotated[:,0], odom_pos_rotated[:,1], 'b:')

plt.legend(['Beacons', 'SLAM', 'Odometry'], loc='upper left')

plt.xlim([-6, None])
plt.axis('equal')
plt.show()
