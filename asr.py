# ╔═══╗  ╔╗          ╔═══╗            ╔═══╗                        ╔╗  
# ║╔═╗║  ║║          ║╔═╗║            ║╔═╗║                        ║║  
# ║║ ║║╔═╝║╔╗╔╗      ║╚══╗╔══╗╔╗      ║╚═╝║╔══╗╔══╗╔══╗╔══╗ ╔═╗╔══╗║╚═╗
# ║╚═╝║║╔╗║║╚╝║      ╚══╗║║╔═╝╠╣      ║╔╗╔╝║╔╗║║══╣║╔╗║╚ ╗║ ║╔╝║╔═╝║╔╗║
# ║╔═╗║║╚╝║╚╗╔╝╔╗    ║╚═╝║║╚═╗║║╔╗    ║║║╚╗║║═╣╠══║║║═╣║╚╝╚╗║║ ║╚═╗║║║║
# ╚╝ ╚╝╚══╝ ╚╝ ╚╝    ╚═══╝╚══╝╚╝╚╝    ╚╝╚═╝╚══╝╚══╝╚══╝╚═══╝╚╝ ╚══╝╚╝╚╝
# Python N-Body Simulation, © 2022 Ryland Goldman
# Using CUDA to Increase the Accuracy and Performance of Particle-Particle N-Body Simulations
# Synopsys Research Project, Los Gatos High School


# NOTE:
# When `main` is called, `NumPy.vectorize` throws a RuntimeWarning "invalid value encountered in double_scalars."
# This warning can be ignored. It is the result of calling `math.asin(dy/r)` when r is zero, which would cause a
# ZeroDivisionError if the function was not vectorized. Due to the use of `NumPy.where`, it doesn't affect the
# program.                                                           

import numpy as np
import math

# constants, 1 now because units don't exist when there's nothing to compare them to (time/distance/speed/direction/etc. are relative)
G = 1      # gravitational constant
k = 1      # coulomb constant
E = 1e-100 # softening constant
t = 1e-7   # time constant
p = 10     # particles
 
# initial conditions
iterations = int(1000) # iterations of simulation
frequency = 1e2      # frequency of recording frames

# data storage, numpy arrays for each of the eight data points
px = np.random.rand(p)    # x, y, z coordinates
py = np.random.rand(p)    # x, y, z coordinates
pz = np.random.rand(p)    # x, y, z coordinates
pvx = np.random.rand(p)*t # component velocities: x, y, z
pvy = np.random.rand(p)*t # component velocities: x, y, z
pvz = np.random.rand(p)*t # component velocities: x, y, z
pq = np.random.rand(p)    # charge
pm = np.random.rand(p)    # mass

# function to calculate the gross acceleration of one particle from another, given the properties of each
# p1x, p1y, p1z    - x, y, and z coordinates of particle 1
# p1vx, p1vy, p1vz - x, y, and z component velocities of particle 1
# p2x, p2y, p2z    - x, y, and z coordinates of particle 2
# p1m, p2m         - masses of particles 1 and 2
# p1q, p2q         - charges of particles 1 and 2
def getForceNV(p1x, p1y, p1z, p1vx, p1vy, p1vz, p1m, p1q, p2x, p2y, p2z, p2m, p2q):
    dx = p1x-p2x # distances between particles in each direction
    dy = p1y-p2y
    dz = p1z-p2z
    r = math.sqrt( dx**2 + dy**2 + dz**2 ) # distance formula
    return t*(( # multiply by time constant
        np.where(
            (p1x == p2x) & (p1y == p2y) & (p1z == p2z), 0.0, # if the particles are the same, then there is no force between them to be calculated
            (G*p1m*p2m)/((r+E)**2) - (k*p1q*p2q)/((r+E)**2)) # otherwise, use newton's law of universal gravitation, and coulomb's law (subtraction because opposites attract, like charges repel)
        )*1.0)/p1m # divide by mass because of newton's 2nd law, so technically it's returning acceleration, not mass

# function to calculate the change in velocity in the x direction
# f                - acceleration provided by getForce function
# p1x, p1y, p1z    - x, y, and z coordinates of particle 1
# p1vx, p1vy, p1vz - x, y, and z component velocities of particle 1
# p2x, p2y, p2z    - x, y, and z coordinates of particle 2
# p1m, p2m         - masses of particles 1 and 2
# p1q, p2q         - charges of particles 1 and 2
def xcompNV(f, p1x, p1y, p1z, p1vx, p1vy, p1vz, p1m, p1q, p2x, p2y, p2z, p2m, p2q):
    dx = p1x-p2x # distances between particles in each direction
    dy = p1y-p2y
    dz = p1z-p2z
    r = math.sqrt( dx**2 + dy**2 + dz**2 ) # distance formula
    alpha = (np.where(dx < 0, -math.asin(dy/r), math.asin(dy/r)))*1.0 # see https://bit.ly/3Hq4s7v - the angle is negative if the x value moves in the negative direction
    beta = (np.where(dz == 0, math.pi, math.atan(dx/dz)))*1.0         # see https://bit.ly/3Hq4s7v - the angle is pi if there is no change in z
    return np.where(f==0, 0, f*math.cos(alpha)*math.sin(beta))*1.0    # see https://bit.ly/3Hq4s7v - if force is zero, no change in x

# function to calculate the change in velocity in the y direction
# f                - acceleration provided by getForce function
# p1x, p1y, p1z    - x, y, and z coordinates of particle 1
# p1vx, p1vy, p1vz - x, y, and z component velocities of particle 1
# p2x, p2y, p2z    - x, y, and z coordinates of particle 2
# p1m, p2m         - masses of particles 1 and 2
# p1q, p2q         - charges of particles 1 and 2
def ycompNV(f, p1x, p1y, p1z, p1vx, p1vy, p1vz, p1m, p1q, p2x, p2y, p2z, p2m, p2q):
    dx = p1x-p2x # distances between particles in each direction
    dy = p1y-p2y
    dz = p1z-p2z
    r = math.sqrt( dx**2 + dy**2 + dz**2 ) # distance formula
    alpha = (np.where(dx < 0, -math.asin(dy/r), math.asin(dy/r)))*1.0 # see https://bit.ly/3Hq4s7v - the angle is negative if the x value moves in the negative direction
    return np.where(f==0, 0, f*math.sin(alpha))*1.0                   # see https://bit.ly/3Hq4s7v - if force is zero, no change in y

# function to calculate the change in velocity in the z direction
# f                - acceleration provided by getForce function
# p1x, p1y, p1z    - x, y, and z coordinates of particle 1
# p1vx, p1vy, p1vz - x, y, and z component velocities of particle 1
# p2x, p2y, p2z    - x, y, and z coordinates of particle 2
# p1m, p2m         - masses of particles 1 and 2
# p1q, p2q         - charges of particles 1 and 2
def zcompNV(f, p1x, p1y, p1z, p1vx, p1vy, p1vz, p1m, p1q, p2x, p2y, p2z, p2m, p2q):
    dx = p1x-p2x # distances between particles in each direction
    dy = p1y-p2y
    dz = p1z-p2z
    r = math.sqrt( dx**2 + dy**2 + dz**2 ) # distance formula
    alpha = (np.where(dx < 0, -math.asin(dy/r), math.asin(dy/r)))*1.0 # see https://bit.ly/3Hq4s7v - the angle is negative if the x value moves in the negative direction
    beta = (np.where(dz == 0, math.pi, math.atan(dx/dz)))*1.0         # see https://bit.ly/3Hq4s7v - the angle is pi if there is no change in z
    return np.where(f==0, 0, f*math.cos(alpha)*math.cos(beta))*1.0    # see https://bit.ly/3Hq4s7v - if force is zero, no change in z

# vectorize functions
getForce = np.vectorize(getForceNV)
xcomp    = np.vectorize(xcompNV)
ycomp    = np.vectorize(ycompNV)
zcomp    = np.vectorize(zcompNV)

# main program function
def main():
    global px, py, pz, pvx, pvy, pvz, pq, pm # global variables
    for n in range(iterations):
        for cp in range(p): # calculate forces on each particle
            forces = getForce( px[cp], py[cp], pz[cp], pvx[cp], pvy[cp], pvz[cp], pm[cp], pq[cp], px, py, pz, pm, pq ) # get acceleration
            chg_vx = xcomp(forces, px[cp], py[cp], pz[cp], pvx[cp], pvy[cp], pvz[cp], pm[cp], pq[cp], px, py, pz, pm, pq ) # change in x velocity
            chg_vy = ycomp(forces, px[cp], py[cp], pz[cp], pvx[cp], pvy[cp], pvz[cp], pm[cp], pq[cp], px, py, pz, pm, pq ) # change in y velocity
            chg_vz = zcomp(forces, px[cp], py[cp], pz[cp], pvx[cp], pvy[cp], pvz[cp], pm[cp], pq[cp], px, py, pz, pm, pq ) # change in z velocity

            # update variables
            pvx[cp] = np.sum(chg_vx)+pvx[cp]
            pvy[cp] = np.sum(chg_vy)+pvy[cp]
            pvz[cp] = np.sum(chg_vz)+pvz[cp]

        # push particles with new velocities
        px += pvx
        py += pvy
        pz += pvz
