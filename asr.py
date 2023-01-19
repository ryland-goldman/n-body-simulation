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
# Copyright © 2022-23 Ryland Goldman
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

FRAMEWORK = "PyOpenCL"
DISPLAY = "Plot"

######## LIBRARIES ########
import os
import sys
import math
import time
import numba
import pandas
import plotly.express as plotlyx
if FRAMEWORK == "CuPy":
    import cupy as np     # import CuPy library
elif FRAMEWORK == "NumPy" or FRAMEWORK == "NumPy-M":
    import numpy as np    # import NumPy library
elif FRAMEWORK == "PyOpenCL":
    import numpy as np    # import NumPy library
    import pyopencl as cl # import PyOpenCL library
elif FRAMEWORK == "None":
    raise NotImplementedError("A framework is currently required.")
else:
    raise RuntimeError("Please specify a valid framework.")

######## CONSTANTS ########
G = 3000.0                 # gravitational constant
k = 0.0                    # coloumb's constant
E = sys.float_info.min     # softening constant
t = 1e-4                   # time constant
p = int(2)               # particles
s = 0.05                   # particle size

######## DATA STORAGE ########
iterations = int(10)      # iterations of simulation
frequency  = int(1)      # frequency of recording frames
px = np.random.rand(p)*7e2 # x, y, z coordinates
py = np.random.rand(p)*7e2 # x, y, z coordinates
pz = np.random.rand(p)*7e2 # x, y, z coordinates
pvx = np.random.rand(p)*t*1e2# component velocities: x, y, z
pvy = np.random.rand(p)*t*1e2# component velocities: x, y, z
pvz = np.random.rand(p)*t*1e2# component velocities: x, y, z
pq = np.ones(p)            # charge
pm = np.ones(p)            # mass
end_process = []           # list to store data which will be processed at the end

######## OPENCL SETUP ########
if FRAMEWORK == "PyOpenCL":
    if E < pow(2,-129): E = pow(2,-129)
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags
    prg = cl.Program(ctx, """
__kernel void force(
        __global const float *p1x_ptr,
        __global const float *p1y_ptr,
        __global const float *p1z_ptr,
        __global const float *p1vxi_ptr,
        __global const float *p1vyi_ptr,
        __global const float *p1vzi_ptr,
        __global const float *p1m_ptr,
        __global const float *p1q_ptr,
        __global const float *p2x,
        __global const float *p2y,
        __global const float *p2z,
        __global const float *p2m,
        __global const float *p2q,
        __global float *p1vx,
        __global float *p1vy,
        __global float *p1vz,
        __global const float *p2vx,
        __global const float *p2vy,
        __global const float *p2vz,
        __global float *v1x,
        __global float *v1y,
        __global float *v1z
    ){
        float p1x = p1x_ptr[0];
        float p1y = p1y_ptr[0];
        float p1z = p1z_ptr[0];
        float p1vxi = p1vxi_ptr[0];
        float p1vyi = p1vyi_ptr[0];
        float p1vzi = p1vzi_ptr[0];
        float p1m = p1m_ptr[0];
        float p1q = p1q_ptr[0];
        int tid = get_global_id(0);
            
        float G = """+f'{G:.20f}'+""";
        float k = """+f'{k:.20f}'+""";
        float E = """+f'{E:.400f}'+""";
        float t = """+f'{t:.200f}'+""";
        float s = """+f'{s:.10f}'+""";

        float dx = p1x - p2x[tid];
        float dy = p1y - p2y[tid];
        float dz = p1z - p2z[tid];
            
        float r = sqrt( dx*dx + dy*dy + dz*dz );
        printf("Hello!");
        if( r != 0.0 ){
            float f = t * (G * p1m * p2m[tid] - k * p1q * p2q[tid])/((r * r+E)*p1m);
            float alpha = asin(dy/(r+E));
            float beta = atan(dx/(dz+E));

            if(dx<0){ alpha = -alpha; }
            
            p1vx[tid] = f * cos(alpha) * sin(beta);
            p1vy[tid] = f * sin(alpha);
            p1vz[tid] = f * cos(alpha) * cos(beta);
                
            if(r < s){
                v1x[tid] = ((p1m - p2m[tid]) * p1vxi + 2 * p2m[tid] * p2vx[tid]) / (p1m + p2m[tid]);
                v1y[tid] = ((p1m - p2m[tid]) * p1vyi + 2 * p2m[tid] * p2vy[tid]) / (p1m + p2m[tid]);
                v1z[tid] = ((p1m - p2m[tid]) * p1vzi + 2 * p2m[tid] * p2vz[tid]) / (p1m + p2m[tid]);
            } else {
                v1x[tid] = 0.0;
                v1y[tid] = 0.0;
                v1z[tid] = 0.0;
            }
        } else {
            p1vx[tid] = 0.0;
            p1vy[tid] = 0.0;
            p1vz[tid] = 0.0;
        }
    }""").build()


######## CUDA SETUP ########
if FRAMEWORK == "CuPy":
    num_blocks = 10
    num_threads = 1000

    if p > (num_blocks * num_threads):
        raise RuntimeError("Invalid number of blocks and threads.")

    force_kernel = np.RawKernel(
        r'''#include <cuda_runtime.h>
        extern "C" __global__
        void force_kernel(
            double p1x, double p1y, double p1z, double p1vxi, double p1vyi, double p1vzi, double p1m, double p1q, double* p2x, double* p2y, double* p2z, double* p2m, double* p2q, double* p1vx, double* p1vy, double* p1vz, double* p2vx, double* p2vy, double* p2vz, double* v1x, double* v1y, double* v1z
        ) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            
            double G = '''+f'{G:.20f}'+''';
            double k = '''+f'{k:.20f}'+''';
            double E = '''+f'{E:.400f}'+''';
            double t = '''+f'{t:.200f}'+''';
            double s = '''+f'{s:.10f}'+''';

            double dx = p1x - p2x[tid];
            double dy = p1y - p2y[tid];
            double dz = p1z - p2z[tid];
            
            double r = sqrt( dx*dx + dy*dy + dz*dz );
            if( r != 0.0 ){
                double f = t * (G * p1m * p2m[tid] - k * p1q * p2q[tid])/((r * r+E)*p1m);
                double alpha = asin(dy/(r+E));
                double beta = atan(dx/(dz+E));

                if(dx<0){ alpha = -alpha; }

                p1vx[tid] = f * cos(alpha) * sin(beta);
                p1vy[tid] = f * sin(alpha);
                p1vz[tid] = f * cos(alpha) * cos(beta);
                
                if(r < s){
                    v1x[tid] = ((p1m - p2m[tid]) * p1vxi + 2 * p2m[tid] * p2vx[tid]) / (p1m + p2m[tid]);
                    v1y[tid] = ((p1m - p2m[tid]) * p1vyi + 2 * p2m[tid] * p2vy[tid]) / (p1m + p2m[tid]);
                    v1z[tid] = ((p1m - p2m[tid]) * p1vzi + 2 * p2m[tid] * p2vz[tid]) / (p1m + p2m[tid]);
                } else {
                    v1x[tid] = 0.0;
                    v1y[tid] = 0.0;
                    v1z[tid] = 0.0;
                }
            } else {
                p1vx[tid] = 0.0;
                p1vy[tid] = 0.0;
                p1vz[tid] = 0.0;
            }
        }''', 'force_kernel')

######## NUMPY SETUP ########
# function to calculate the acceleration of one particle on another given the distance, mass, and charge
# returns a tuple of the component forces and (if collision) velocity, in the format of (x,y,z,vx,vy,vz)
# p1x, p1y, p1z    - x, y, and z coordinates of particle 1
# p1vx, p1vy, p1vz - x, y, and z component velocities of particle 1
# p2x, p2y, p2z    - x, y, and z coordinates of particle 2
# p1m, p2m         - masses of particles 1 and 2
# p1q, p2q         - charges of particles 1 and 2
# p2vx, p2vy, p2vz - x, y, and z component velocities of particle 2
@numba.njit(error_model="numpy", parallel=(FRAMEWORK=="NumPy-M"), fastmath=True, cache=True, nogil=True)
def getForceNV(p1x, p1y, p1z, p1vx, p1vy, p1vz, p1m, p1q, p2x, p2y, p2z, p2m, p2q , p2vx, p2vy, p2vz):
    dx = p1x-p2x # distances between particles in each direction
    dy = p1y-p2y
    dz = p1z-p2z

    # distance formula
    r = dx**2 + dy**2 + dz**2
    r = np.sqrt( r )

    # calculate force
    f = t * (G*p1m*p2m - k*p1q*p2q)/( (E + r**2 ) * p1m) # use newton's law of universal gravitation, and coulomb's law (subtraction because opposites attract, like charges repel), divide by mass because of newton's 2nd law

    # calculate angles - see https://bit.ly/3Hq4s7v
    alpha = np.where(dx < 0, -np.arcsin(dy/(r+E)), np.arcsin(dy/(r+E))) # the angle is negative if the x value moves in the negative direction
    beta = np.arctan(dx/(dz+E))

    # calculate component vectors of forces - see https://bit.ly/3Hq4s7v
    xforce = np.where(r==0, 0, f*np.cos(alpha)*np.sin(beta))
    yforce = np.where(r==0, 0, f*np.sin(alpha))
    zforce = np.where(r==0, 0, f*np.cos(alpha)*np.cos(beta))

    # check if collision occurs
    v1x = np.where( (r < s) & (r != 0), (((p1m - p2m) * p1vx + 2 * p2m * p2vx) / (p1m + p2m)), 0)
    v1y = np.where( (r < s) & (r != 0), (((p1m - p2m) * p1vy + 2 * p2m * p2vy) / (p1m + p2m)), 0)
    v1z = np.where( (r < s) & (r != 0), (((p1m - p2m) * p1vz + 2 * p2m * p2vz) / (p1m + p2m)), 0)
    
    return (xforce, yforce, zforce, v1x, v1y, v1z)

if FRAMEWORK == "NumPy" or FRAMEWORK == "NumPy-M":
    getForce = np.vectorize(getForceNV) # vectorize the function

######## MAIN PROGRAM FUNCTION ########
def main():
    global px, py, pz, pvx, pvy, pvz, pq, pm # global variables
    for n in range(iterations):
        if (n/iterations)*100 % 1 == 0 and n != 0:
            now = round(time.time()-start_time,3)
            left = round(now*iterations/n-now,3)
            print((n/iterations)*100,"% complete\tETA:",str(left)+"s remaining ("+str(now)+"s elapsed)")
        
        if n % frequency == 0:
            end_process.append([n, px.tolist(), py.tolist(), pz.tolist()])
        tmp_vx, tmp_vy, tmp_vz = pvx, pvy, pvz
        if FRAMEWORK == "NumPy-M":
            for cp in numba.prange(p):
                chg_vx, chg_vy, chg_vz, cls_vx, cls_vy, cls_vz = getForceNV( px[cp], py[cp], pz[cp], pvx[cp], pvy[cp], pvz[cp], pm[cp], pq[cp], px, py, pz, pm, pq, tmp_vx, tmp_vy, tmp_vz ) # get acceleration

                # update variables
                pvx[cp] = np.sum(chg_vx)+pvx[cp]
                pvy[cp] = np.sum(chg_vy)+pvy[cp]
                pvz[cp] = np.sum(chg_vz)+pvz[cp]

                # if collision, update variables again
                if np.sum(cls_vx) != 0:
                    pvx[cp] = np.sum(cls_vx)
                    pvy[cp] = np.sum(cls_vy)
                    pvz[cp] = np.sum(cls_vz)
                    
            
        else:
            for cp in range(p): # calculate forces on each particle

                if FRAMEWORK == "NumPy":
                    chg_vx, chg_vy, chg_vz, cls_vx, cls_vy, cls_vz = getForce( px[cp], py[cp], pz[cp], pvx[cp], pvy[cp], pvz[cp], pm[cp], pq[cp], px, py, pz, pm, pq, tmp_vx, tmp_vy, tmp_vz ) # get acceleration

                    # update variables
                    pvx[cp] = np.sum(chg_vx)+pvx[cp]
                    pvy[cp] = np.sum(chg_vy)+pvy[cp]
                    pvz[cp] = np.sum(chg_vz)+pvz[cp]

                    # if collision, update variables again
                    if np.sum(cls_vx) != 0:
                        pvx[cp] = np.sum(cls_vx)
                        pvy[cp] = np.sum(cls_vy)
                        pvz[cp] = np.sum(cls_vz)
                if FRAMEWORK == "CuPy":
                    chg_vx = np.zeros((p))
                    chg_vy = np.zeros((p))
                    chg_vz = np.zeros((p))
                    cls_vx = np.zeros((p))
                    cls_vy = np.zeros((p))
                    cls_vz = np.zeros((p))
                    
                    force_kernel((num_blocks,),(num_threads,),(float(px[cp]), float(py[cp]), float(pz[cp]), float(tmp_vx[cp]), float(tmp_vy[cp]), float(tmp_vz[cp]), float(pm[cp]), float(pq[cp]), px, py, pz, pm, pq, chg_vx, chg_vy, chg_vz, tmp_vx, tmp_vy, tmp_vz, cls_vx, cls_vy, cls_vz)) # get acceleration

                    # update variables
                    pvx[cp] = np.sum(chg_vx)+pvx[cp]
                    pvy[cp] = np.sum(chg_vy)+pvy[cp]
                    pvz[cp] = np.sum(chg_vz)+pvz[cp]
                    
                    # if collision, update variables again
                    if np.sum(cls_vx) != 0:
                        pvx[cp] = np.sum(cls_vx)
                        pvy[cp] = np.sum(cls_vy)
                        pvz[cp] = np.sum(cls_vz)
                        
                if FRAMEWORK == "PyOpenCL":

                    # transfer data to opencl
                    px_g =   cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=px)
                    py_g =   cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=py)
                    pz_g =   cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=pz)
                    pvxt_g = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=tmp_vx)
                    pvyt_g = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=tmp_vy)
                    pvzt_g = cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=tmp_vz)
                    pm_g =   cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=pm)
                    pq_g =   cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=pq)
                    chg_vxg =cl.Buffer(ctx, mf.WRITE_ONLY, pvx.nbytes)
                    chg_vyg =cl.Buffer(ctx, mf.WRITE_ONLY, pvy.nbytes)
                    chg_vzg =cl.Buffer(ctx, mf.WRITE_ONLY, pvz.nbytes)
                    cls_vxg =cl.Buffer(ctx, mf.WRITE_ONLY, pvx.nbytes)
                    cls_vyg =cl.Buffer(ctx, mf.WRITE_ONLY, pvy.nbytes)
                    cls_vzg =cl.Buffer(ctx, mf.WRITE_ONLY, pvz.nbytes)
                    chg_vx =  np.empty_like(pvx)
                    chg_vy =  np.empty_like(pvy)
                    chg_vz =  np.empty_like(pvz)
                    cls_vx =  np.empty_like(pvx)
                    cls_vy =  np.empty_like(pvy)
                    cls_vz =  np.empty_like(pvz)

                    # calculate acceleration
                    prg.force(queue, px.shape, None,
                              cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=np.array(float(px[cp]))),
                              cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=np.array(float(py[cp]))),
                              cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=np.array(float(pz[cp]))),
                              cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=np.array(float(tmp_vx[cp]))),
                              cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=np.array(float(tmp_vy[cp]))),
                              cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=np.array(float(tmp_vz[cp]))),
                              cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=np.array(float(pm[cp]))),
                              cl.Buffer(ctx, mf.READ_ONLY| mf.COPY_HOST_PTR, hostbuf=np.array(float(pq[cp]))),
                              px_g, py_g, pz_g, pm_g, pq_g, chg_vxg, chg_vyg, chg_vzg, pvxt_g, pvyt_g, pvzt_g, cls_vxg, cls_vyg, cls_vzg)

                    # copy data to cpu
                    cl.enqueue_copy(queue, chg_vx, chg_vxg)
                    cl.enqueue_copy(queue, chg_vy, chg_vyg)
                    cl.enqueue_copy(queue, chg_vz, chg_vzg)
                    cl.enqueue_copy(queue, cls_vx, cls_vxg)
                    cl.enqueue_copy(queue, cls_vy, cls_vyg)
                    cl.enqueue_copy(queue, cls_vz, cls_vzg)
                    
                    # update variables
                    pvx[cp] = np.sum(chg_vx)+pvx[cp]
                    pvy[cp] = np.sum(chg_vy)+pvy[cp]
                    pvz[cp] = np.sum(chg_vz)+pvz[cp]
                    
                    # if collision, update variables again
                    if np.sum(cls_vx) != 0:
                        pvx[cp] = np.sum(cls_vx)
                        pvy[cp] = np.sum(cls_vy)
                        pvz[cp] = np.sum(cls_vz)
    
        # push particles with new velocities
        px += pvx
        py += pvy
        pz += pvz


######## VIDEO PROCESSING ########
# function to convert a 2D array into a video animation
# frames - 2D array, with the first dimension representing individual frames, and the second dimension containing a counter and lists of x, y, and z coordinates
def create_video(frames):
    print("100.0% complete  \tProcessing...")
    if DISPLAY == 'Video' or DISPLAY == 'Both':
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
    if DISPLAY == 'Plot' or DISPLAY == 'Both':
        data_x = []
        data_y = []
        data_z = []
        data_p = []
        data_f = []
        for frame in frames:
            for p in range(len(frame[1])):
                data_x.append(frame[1][p])
                data_y.append(frame[2][p])
                data_z.append(frame[3][p])
                data_p.append(p)
                data_f.append(frame[0])
        data = pandas.DataFrame(data={'x':data_x,'y':data_y,'z':data_z,'f':data_f,'p':data_p})
        fig = plotlyx.scatter_3d(data, x='x', y='y', z='z', animation_frame='f', animation_group='p')
        fig.update_layout(scene=dict(xaxis=dict(range=[min(data_x), max(data_x)],autorange=False),yaxis=dict(range=[min(data_y), max(data_y)],autorange=False),zaxis=dict(range=[min(data_z), max(data_z)],autorange=False)))
        fig.show()

###### COMPILE #####
print("Compiling...")
precompile_time = time.time() # time before compilation
if FRAMEWORK == "NumPy-M": tcvx, tcvy, tcvz, tclsvx, tclsvy, tclsvz = getForceNV( px[0], py[0], pz[0], pvx[0], pvy[0], pvz[0], pm[0], pq[0], px, py, pz, pm, pq, pvx, pvy, pvz )
if FRAMEWORK == "NumPy": tcvx, tcvy, tcvz, tclsvx, tclsvy, tclsvz = getForce( px[0], py[0], pz[0], pvx[0], pvy[0], pvz[0], pm[0], pq[0], px, py, pz, pm, pq, pvx, pvy, pvz )
if FRAMEWORK == "CuPy":
    tchg_vx = np.zeros((p))
    tchg_vy = np.zeros((p))
    tchg_vz = np.zeros((p))
    tcls_vx = np.zeros((p))
    tcls_vy = np.zeros((p))
    tcls_vz = np.zeros((p))       
    force_kernel((num_blocks,),(num_threads,),(float(px[0]), float(py[0]), float(pz[0]), float(pvx[0]), float(pvy[0]), float(pvz[0]), float(pm[0]), float(pq[0]), px, py, pz, pm, pq, tchg_vx, tchg_vy, tchg_vz, pvx, pvy, pvz, tcls_vx, tcls_vy, tcls_vz))

print("Beginning N-body simulation")
start_time = time.time()    # start of program
main()                      # run program
midpoint_time = time.time() # runtime of program, exluding animation
create_video(end_process)   # create animation
end_time = time.time()      # runtime of program, including animation

print("Program has completed running using",FRAMEWORK)
if FRAMEWORK == "CuPy": print("Blocks/Threads:",num_blocks,"x",num_threads)
print(p,"particles for",iterations,"frames, recording every",frequency,"frames")
print("Time to compile functions:     ",math.floor((start_time-precompile_time)*100)/100," seconds")
print("Time to run N-body simulation: ",math.floor((midpoint_time-start_time)*100)/100," seconds")
print("Time to create animation:      ",math.floor((end_time-midpoint_time)*100)/100," seconds")
print("Total time:                    ",math.floor((end_time-precompile_time)*100)/100," seconds")
