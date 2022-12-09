import numpy as np
import math

# constants, 1 now because units don't exist when there's nothing to compare them to (time/distance/speed/direction/etc. are relative)
G = 1  # gravitational constant
k = 1 # coulomb constant
E = 1e-100 # softening constant
t = 1e-7 # time constant
s = 1 # size constant
p = 10 # particles
 
# initial conditions
iterations = int(10) # iterations of simulation
frequency = 1e2 # frequency of recording frames

# data storage
particles = np.random.rand(p, 8)
# 2D array to store ten particles with the following eight datapoints:
#   X, Y, Z, Vx, Vy, Vz, Q, M
#   0, 1, 2,  3,  4,  5, 6, 7

def ParticleIntrNV(p1, p2):
    print(p1)
    # don't calculate forces on the same particle
    if p1 == p2:
        return p1
    
    # calculate force using distance formula: sqrt [ (x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2 ]
    r = math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2 )

    # calculate acceeleration by dividing force by mass
    #    gravitational force (Gmm/r^2) plus electromagnetic force (kqq/r^2)
    a = (G*p1[7]*p2[7]/((r+E)**2)) + (-k*p1[6]*p2[6]/((r+E)**2))

    # differences
    dx = p1[0]-p2[0]
    dy = p1[1]-p2[1]
    dz = p1[2]-p2[2]

    # calculate angles
    try:
        alpha = math.asin(dy/r)
    except ValueError:
        if dy/r == 0:
            alpha = math.pi/2
        else:
            alpha = -math.pi/2
    if dz == 0:
        beta = math.pi
    else:
        beta = math.atan(dx/dz)
    if dx < 0: alpha = -alpha - math.pi

    # convert to component vectors, multiply by time constant, add
    return [p1[0], p1[1], p1[2], p1[3]+a*math.cos(alpha)*math.sin(beta)*t, p1[4]+a*math.sin(alpha)*t, p1[5]+a*math.cos(alpha)*math.cos(beta)*t, p1[6], p1[7]]

ParticleIntr = np.vectorize(ParticleIntrNV)

def main():
    global particles
    for n in range(iterations):
        for cp in range(p):
            particles[cp] = ParticleIntr(particles, particles[cp])
