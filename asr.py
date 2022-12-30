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

FRAMEWORK = "CuPy" # set to `CuPy`, `NumPy`, or `PyOpenCL`

######## LIBRARIES ########
import os
import sys
import math
import time
if FRAMEWORK == "CuPy":
    import cupy as np     # import CuPy library
elif FRAMEWORK == "NumPy":
    import numpy as np    # import NumPy library
    import numba
elif FRAMEWORK == "PyOpenCL":
    raise NotImplementedError("PyOpenCL has not yet been implemented")
    import pyopencl as cl # import PyOpenCL library
elif FRAMEWORK == "None":
    raise NotImplementedError("A framework is currently required.")
else:
    raise RuntimeError("Please specify a valid framework.")

######## CONSTANTS ########
G = 1.0                   # gravitational constant
k = 1.0                   # coloumb's constant
E = sys.float_info.min    # softening constant
t = 1e-3                  # time constant
p = int(1e2)              # particles

######## DATA STORAGE ########
iterations = int(1e2)     # iterations of simulation
frequency  = int(1)      # frequency of recording frames
px = np.random.rand(p)    # x, y, z coordinates
py = np.random.rand(p)    # x, y, z coordinates
pz = np.random.rand(p)    # x, y, z coordinates
pvx = np.random.rand(p)*t # component velocities: x, y, z
pvy = np.random.rand(p)*t # component velocities: x, y, z
pvz = np.random.rand(p)*t # component velocities: x, y, z
pq = np.random.rand(p)    # charge
pm = np.random.rand(p)    # mass
end_process = []          # list to store data which will be processed at the end

######## CUDA SETUP ########
if FRAMEWORK == "CuPy":
    num_blocks = 10
    num_threads = 10

    force_kernel = np.RawKernel(
        r'''
        extern "C" __global__
        void force_kernel(
            const double p1x, const double p1y, const double p1z, const double p1m, const double p1q, const double* p2x, const double* p2y, const double* p2z, const double* p2m, const double* p2q, double* p1vx, double* p1vy, double* p1vz
        ) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;

            double G = '''+f'{G:.20f}'+''';
            double k = '''+f'{k:.20f}'+''';
            double E = '''+f'{E:.400f}'+''';
            double t = '''+f'{t:.200f}'+''';

            double dx = p1x-p2x[tid];
            double dy = p1y-p2y[tid];
            double dz = p1z-p2z[tid];
            
            float r = sqrt( dx*dx + dy*dy + dz*dz );
            if( r != 0.0 ){
                double f = t*(G*p1m*p2m[tid] - k*p1q*p2q[tid])/(r*r+E);
                double alpha = asin(dy/(r+E));
                double beta = atan(dx/(dz+E));

                if(dx<0){ alpha = -alpha; }

                p1vx[tid] = f * cos(alpha) * sin(beta);
                p1vy[tid] = f * sin(alpha);
                p1vz[tid] = f * cos(alpha) * cos(beta);
            } else {
                p1vx[tid] = 0.0;
                p1vy[tid] = 0.0;
                p1vz[tid] = 0.0;
            }
        }''', 'force_kernel')

######## NUMPY SETUP ########
# function to calculate the acceleration of one particle on another given the distance, mass, and charge
# returns a tuple of the component forces, in the format of (x,y,z)
# p1x, p1y, p1z    - x, y, and z coordinates of particle 1
# p1vx, p1vy, p1vz - x, y, and z component velocities of particle 1
# p2x, p2y, p2z    - x, y, and z coordinates of particle 2
# p1m, p2m         - masses of particles 1 and 2
# p1q, p2q         - charges of particles 1 and 2
# @numba.njit(parallel=True)
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
if FRAMEWORK == "NumPy":
    getForce = np.vectorize(getForceNV) # vectorize the function


######## MAIN PROGRAM FUNCTION ########
def main():
    global px, py, pz, pvx, pvy, pvz, pq, pm # global variables
    for n in range(iterations):
        if n % frequency == 0:
            end_process.append([n, px.tolist(), py.tolist(), pz.tolist()])
        
        for cp in range(p): # calculate forces on each particle

            if FRAMEWORK == "NumPy":
                chg_vx, chg_vy, chg_vz = getForce( px[cp], py[cp], pz[cp], pvx[cp], pvy[cp], pvz[cp], pm[cp], pq[cp], px, py, pz, pm, pq ) # get acceleration

                # update variables
                pvx[cp] = np.sum(chg_vx)+pvx[cp]
                pvy[cp] = np.sum(chg_vy)+pvy[cp]
                pvz[cp] = np.sum(chg_vz)+pvz[cp]

            if FRAMEWORK == "CuPy":
                chg_vx = np.zeros((p))
                chg_vy = np.zeros((p))
                chg_vz = np.zeros((p))
                force_kernel((num_blocks,),(num_threads,),(px[cp], py[cp], pz[cp], pm[cp], pq[cp], px, py, pz, pm, pq, chg_vx, chg_vy, chg_vz)) # get acceleration

                # update variables
                pvx[cp] = np.sum(chg_vx)+pvx[cp]
                pvy[cp] = np.sum(chg_vy)+pvy[cp]
                pvz[cp] = np.sum(chg_vz)+pvz[cp]

        # push particles with new velocities
        px += pvx
        py += pvy
        pz += pvz


######## VIDEO PROCESSING ########
# function to convert a 2D array into a video animation
# frames - 2D array, with the first dimension representing individual frames, and the second dimension containing a counter and lists of x, y, and z coordinates
def create_video(frames):
    import matplotlib as mp
    mp.use("agg") # alternatively, set backend of matplotlib to Tkinter ("TkAgg")
    import matplotlib.pyplot as plt
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

print("Program has completed running using",FRAMEWORK)
if FRAMEWORK == "CuPy":
    print(p," particles at ",num_blocks,"x",num_threads)
print("Time to run N-body simulation: ",math.floor((midpoint_time-start_time)*100)/100," seconds")
print("Time to create animation:      ",math.floor((end_time-midpoint_time)*100)/100," seconds")
print("Total time:                    ",math.floor((end_time-start_time)*100)/100," seconds")
