# ╔═══╗  ╔╗          ╔═══╗            ╔═══╗                        ╔╗  
# ║╔═╗║  ║║          ║╔═╗║            ║╔═╗║                        ║║  
# ║║ ║║╔═╝║╔╗╔╗      ║╚══╗╔══╗╔╗      ║╚═╝║╔══╗╔══╗╔══╗╔══╗ ╔═╗╔══╗║╚═╗
# ║╚═╝║║╔╗║║╚╝║      ╚══╗║║╔═╝╠╣      ║╔╗╔╝║╔╗║║══╣║╔╗║╚ ╗║ ║╔╝║╔═╝║╔╗║
# ║╔═╗║║╚╝║╚╗╔╝╔╗    ║╚═╝║║╚═╗║║╔╗    ║║║╚╗║║═╣╠══║║║═╣║╚╝╚╗║║ ║╚═╗║║║║
# ╚╝ ╚╝╚══╝ ╚╝ ╚╝    ╚═══╝╚══╝╚╝╚╝    ╚╝╚═╝╚══╝╚══╝╚══╝╚═══╝╚╝ ╚══╝╚╝╚╝
# Python N-Body Simulation
# Using CUDA to Increase the Accuracy and Performance of Particle-Particle N-Body Simulations
# Synopsys Research Project, Los Gatos High School


# MIT License
#
# Copyright © 2022 Ryland Goldman
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# NOTE:
# When `main` is called, `NumPy.vectorize` throws a RuntimeWarning "invalid value encountered in double_scalars."
# This warning can be ignored. It is the result of calling `math.asin(dy/r)` when r is zero, which would cause a
# ZeroDivisionError if the function was not vectorized. Due to the use of `NumPy.where`, it doesn't affect the
# program.                                                           

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import os
import math

# constants, 1 now because units don't exist when there's nothing to compare them to (time/distance/speed/direction/etc. are relative)
G = 1      # gravitational constant
k = 1      # coulomb constant
E = 1e-100 # softening constant
t = 1e-7   # time constant
p = 10     # particles
 
# initial conditions
iterations = int(1000) # iterations of simulation
frequency  = int(10)  # frequency of recording frames

# data storage, numpy arrays for each of the eight data points
px = np.random.rand(p)    # x, y, z coordinates
py = np.random.rand(p)    # x, y, z coordinates
pz = np.random.rand(p)    # x, y, z coordinates
pvx = np.random.rand(p)*t # component velocities: x, y, z
pvy = np.random.rand(p)*t # component velocities: x, y, z
pvz = np.random.rand(p)*t # component velocities: x, y, z
pq = np.random.rand(p)    # charge
pm = np.random.rand(p)    # mass
end_process = []          # list to store data which will be processed at the end

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
        if n % frequency == 0:
            end_process.append([n, px.tolist(), py.tolist(), pz.tolist()])
        
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

# function to convert a 2D array into a video animation
# frames - 2D array, with the first dimension representing individual frames, and the second dimension containing a counter and lists of x, y, and z coordinates
def create_video(frames):
    mp.use("TkAgg") # set backend of matplotlib to Tkinter
    counter = 0     # create a counter
    for frame in frames:   # loop through each frame in list
        fig = plt.figure() # create a new plot
        ax = fig.add_subplot(projection='3d')    # new 3D plot
        ax.clear()         # clear plot
        ax.scatter3D(frame[1],frame[2],frame[3]) # add x, y, and z axes
        plt.savefig('C:\\Nbody\\files\\frame-'+str(counter)+'.png') # save image in C:\Nbody
        ax.clear()         # clear plot
        plt.close(fig)     # close plot
        counter = counter + 1 # increment counter

    # use FFmpeg to generate a video, switching the frame rate based on the number of frames
    if iterations/frequency > 2500:
        os.system("C:\\Nbody\\ffmpeg.exe -f image2 -r 60 -i C:\\Nbody\\files\\frame-%01d.png -vcodec mpeg4 -y C:\\Nbody\\video.mp4")
    if iterations/frequency > 500:
        os.system("C:\\Nbody\\ffmpeg.exe -f image2 -r 30 -i C:\\Nbody\\files\\frame-%01d.png -vcodec mpeg4 -y C:\\Nbody\\video.mp4")
    if iterations/frequency > 100:
        os.system("C:\\Nbody\\ffmpeg.exe -f image2 -r 20 -i C:\\Nbody\\files\\frame-%01d.png -vcodec mpeg4 -y C:\\Nbody\\video.mp4")
    else:
        os.system("C:\\Nbody\\ffmpeg.exe -f image2 -r 10 -i C:\\Nbody\\files\\frame-%01d.png -vcodec mpeg4 -y C:\\Nbody\\video.mp4")

    # remove all files in directory
    filelist = [ f for f in os.listdir("C:\\Nbody\\files\\") if f.endswith(".png") ]
    for f in filelist:
        os.remove(os.path.join("C:\\Nbody\\files\\", f))
        
main()                    # run program
create_video(end_process) # create animation
