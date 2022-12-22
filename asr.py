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

CUPY_OR_NUMPY = "CuPy"
import matplotlib as mp
import matplotlib.pyplot as plt
import os
import math
import time

if CUPY_OR_NUMPY == "NumPy":
    import numpy as np
elif CUPY_OR_NUMPY == "CuPy":
    import cupy as np
else:
    raise Exception("Please set a library to use")

# constants, 1 now because units don't exist when there's nothing to compare them to (time/distance/speed/direction/etc. are relative)
G = 1      # gravitational constant
k = 1      # coulomb constant
E = 1e-100 # softening constant
t = 1e-7   # time constant
p = 10     # particles
 
# initial conditions
iterations = int(1e3) # iterations of simulation
frequency  = int(5e2)  # frequency of recording frames

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

# function to calculate the acceleration of one particle on another given the distance, mass, and charge
# returns a tuple of the component forces, in the format of (x,y,z)
# p1x, p1y, p1z    - x, y, and z coordinates of particle 1
# p1vx, p1vy, p1vz - x, y, and z component velocities of particle 1
# p2x, p2y, p2z    - x, y, and z coordinates of particle 2
# p1m, p2m         - masses of particles 1 and 2
# p1q, p2q         - charges of particles 1 and 2
def getForceNV(p1x, p1y, p1z, p1vx, p1vy, p1vz, p1m, p1q, p2x, p2y, p2z, p2m, p2q):
    dx = p1x-p2x # distances between particles in each direction
    dy = p1y-p2y
    dz = p1z-p2z

    # distance formula
    r = dx**2 + dy**2 + dz**2
    r = np.sqrt( r ) + E 

    # calculate force
    f = t*(( # multiply by time constant
        np.where(
            (p1x == p2x) & (p1y == p2y) & (p1z == p2z), 0.0, # if the particles are the same, then there is no force between them to be calculated
            (G*p1m*p2m)/((r)**2) - (k*p1q*p2q)/((r)**2))     # otherwise, use newton's law of universal gravitation, and coulomb's law (subtraction because opposites attract, like charges repel)
        )*1.0)/p1m # divide by mass because of newton's 2nd law, so technically it's returning acceleration, not mass

    # calculate angles - see https://bit.ly/3Hq4s7v
    alpha = np.where(dx < 0, -np.arcsin(dy/r), np.arcsin(dy/r)) # the angle is negative if the x value moves in the negative direction
    beta = np.arctan(dx/(dz+E))

    # calculate component vectors of forces - see https://bit.ly/3Hq4s7v
    xforce = np.where(f==0, 0, f*np.cos(alpha)*np.sin(beta))
    yforce = np.where(f==0, 0, f*np.sin(alpha))
    zforce = np.where(f==0, 0, f*np.cos(alpha)*np.cos(beta))
    return (xforce, yforce, zforce)

# vectorize getForce function
getForce = np.vectorize(getForceNV) #np.frompyfunc(getForceNV, 13, 3)

# main program function
def main():
    global px, py, pz, pvx, pvy, pvz, pq, pm # global variables
    for n in range(iterations):
        if n % frequency == 0:
            end_process.append([n, px.tolist(), py.tolist(), pz.tolist()])
        
        for cp in range(p): # calculate forces on each particle
            chg_vx, chg_vy, chg_vz = getForce( px[cp], py[cp], pz[cp], pvx[cp], pvy[cp], pvz[cp], pm[cp], pq[cp], px, py, pz, pm, pq ) # get acceleration

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

start_time = time.time()    # start of program
main()                      # run program
midpoint_time = time.time() # runtime of program, exluding animation
create_video(end_process)   # create animation
end_time = time.time()      # runtime of program, including animation

print("Program has completed running.")
print("Time to run N-body simulation: ",math.floor((midpoint_time-start_time)*100)/100," seconds")
print("Time to create animation:      ",math.floor((end_time-midpoint_time)*100)/100," seconds")
print("Total time:                    ",math.floor((end_time-start_time)*100)/100," seconds")
