import numpy as np

# constants, 1 now because units don't exist when there's nothing to compare them to (time/distance/speed/direction/etc. are relative)
G = 1  # gravitational constant
k = 1 # coulomb constant
E = 1e-100 # softening constant
t = 1e-7 # time constant
s = 1 # size constant
p = 10 # particles
 
# initial conditions
iterations = 5e4 # iterations of simulation
frequency = 1e2 # frequency of recording frames

# data storage
particles = np.zeros((p, 8))
# 2D array to store ten particles with the following eight datapoints:
#   X, Y, Z, Vx, Vy, Vz, Q, M

def particle_intr_L0NV(particle):
    # level zero
PL0 = np.frompyfunc(particle_intr_L0NV)
def particle_intr_L1NV(particle):
    # level one
PL1 = np.frompyfunc(particle_intr_L1NV)

def main():
    global particles
    for n in range(iterations):
        PL0(particles)
