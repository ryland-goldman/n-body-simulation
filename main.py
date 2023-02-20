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

######## GLOBALS ########
# FRAMEWORK options:
#    CuPy         - Use the CuPy library for multithreading with a GPU via CUDA
#    NumPy        - Use the NumPy library for a single-thread on the CPU
#    NumPy-M      - Uses NumPy plus Numba for multithreading on the CPU
#    PyOpenCL     - Any device running OpenCL
#    PyOpenCL-CPU - Forces OpenCL to run on the CPU
#    PyOpenCL-GPU - Forces OpenCL to run on the GPU
# DISPLAY options:
#    None         - No output
#    Plot         - Interactive, animated plot with Plotly (faster)
#    Video        - MP4 video output with Matplotlib/FFmpeg (slower)
#    Both         - Creates a plot and a video
# OUTPUT options:
#    0            - Nothing logged to console
#    1            - Only simulation runtime logged to console
#    2            - Show full end output
#    3            - Show full end output and progress
import sys
if not sys.stdin.isatty():
    FRAMEWORK = sys.argv[1]
    DISPLAY = sys.argv[2]
    OUTPUT = 1
else:
    FRAMEWORK = "CuPy"
    DISPLAY = "None"
    OUTPUT = 3

######## LIBRARIES ########
import os     # For running FFmpeg and saving a generated video to the disk
import time   # For measuring the time
import numba  # For wrapping getForceNV function
import pandas # Required for Plotly
import plotly.express as plotlyx # For plotting
if FRAMEWORK == "CuPy": import cupy as np                                 # import CuPy library if needed
elif FRAMEWORK == "NumPy" or FRAMEWORK == "NumPy-M":  import numpy as np  # import NumPy library if needed
elif FRAMEWORK == "PyOpenCL-CPU" or FRAMEWORK == "PyOpenCL-GPU" or FRAMEWORK == "PyOpenCL":
    import numpy as np                                                    # import NumPy library
    import pyopencl as cl                                                 # import PyOpenCL library
    import pyopencl.array as cl_array                                     # import array from PyOpenCL
else: raise RuntimeError("Please specify a valid framework.")             # if no framework is specified, raise an error

######## CONSTANTS ########
G = 3000.0                 # gravitational constant
k = 0.0                    # coloumb's constant
E = pow(2,1)               # softening constant
t = 1e-2                   # time constant
p = int(4096)              # particles
s = 0.05                   # particle size
if not sys.stdin.isatty(): p = int(sys.argv[4])

######## DATA STORAGE ########
iterations = int(5)          # iterations of simulation
frequency  = int(1)          # frequency of recording frames
px = np.random.rand(p)*7e2   # x, y, z coordinates
py = np.random.rand(p)*7e2   # x, y, z coordinates
pz = np.random.rand(p)*7e2   # x, y, z coordinates
pvx = np.random.rand(p)*t*1e2# component velocities: x, y, z
pvy = np.random.rand(p)*t*1e2# component velocities: x, y, z
pvz = np.random.rand(p)*t*1e2# component velocities: x, y, z
pq = np.ones(p)              # charge
pm = np.ones(p)              # mass
end_process = []             # list to store data which will be processed at the end
if not sys.stdin.isatty(): iterations = int(sys.argv[3])

######## OPENCL SETUP ########
if FRAMEWORK == "PyOpenCL-CPU" or FRAMEWORK == "PyOpenCL-GPU" or FRAMEWORK == "PyOpenCL":
    # for some reason, the opencl setup causes an error
    import warnings
    warnings.filterwarnings("ignore")

    platform = cl.get_platforms()
    
    # Only select GPUs from platform 0
    if FRAMEWORK == "PyOpenCL-GPU":
        devices = platform[0].get_devices(device_type=cl.device_type.GPU)

    # Only select CPUs from platform 1
    elif FRAMEWORK == "PyOpenCL-CPU":
        devices = platform[1].get_devices(device_type=cl.device_type.CPU)

    # FP32 compatibility
    if E < pow(2,-128): E = pow(2,-128)
    
    if FRAMEWORK == "PyOpenCL":
        # User selects device
        ctx = cl.create_some_context()
    else:
        # Use pre-selected device
        ctx = cl.Context(devices=devices)

    queue = cl.CommandQueue(ctx) # OpenCL command queue
    mf = cl.mem_flags            # Memory flags
    constants = r"""double G = """+f'{G:.20f}'+r""";
        double k = """+f'{k:.20f}'+r""";
        double E = """+f'{E:.400f}'+r""";
        double t = """+f'{t:.200f}'+r""";
        double s = """+f'{s:.10f}'+r""";""";
    kernel_import = open("C:\\Nbody\\kernel.cl","r").read().replace("//ImportConstants",constants)
    prg = cl.Program(ctx, kernel_import).build()


######## CUDA SETUP ########
if FRAMEWORK == "CuPy":
    num_blocks = 4
    num_threads = 1024

    constants = r"""double G = """+f'{G:.20f}'+r""";
        double k = """+f'{k:.20f}'+r""";
        double E = """+f'{E:.400f}'+r""";
        double t = """+f'{t:.200f}'+r""";
        double s = """+f'{s:.10f}'+r""";""";
    kernel_import = open("C:\\Nbody\\kernel.cu","r").read().replace("//ImportConstants",constants)
    
    force_kernel = np.RawKernel(kernel_import, 'force_kernel')

######## NUMPY SETUP ########
# function to calculate the acceleration of one particle on another given the distance, mass, and charge
# returns a tuple of the component forces and (if collision) velocity, in the format of (x,y,z,vx,vy,vz)
# p1x, p1y, p1z    - x, y, and z coordinates of particle 1
# p1vx, p1vy, p1vz - x, y, and z component velocities of particle 1
# p2x, p2y, p2z    - x, y, and z coordinates of particle 2
# p1m, p2m         - masses of particles 1 and 2
# p1q, p2q         - charges of particles 1 and 2
# p2vx, p2vy, p2vz - x, y, and z component velocities of particle 2
@numba.njit(error_model="numpy", parallel=(FRAMEWORK=="NumPy-M"), fastmath=True, cache=True)
def getForceNV(p1x, p1y, p1z, p1vx, p1vy, p1vz, p1m, p1q, p2x, p2y, p2z, p2m, p2q , p2vx, p2vy, p2vz):
    dx = p1x-p2x # distances between particles in each direction
    dy = p1y-p2y
    dz = p1z-p2z

    # distance formula
    r = dx**2 + dy**2 + dz**2
    r = np.sqrt( r )

    # calculate force
    f = t * r * (G*p1m*p2m - k*p1q*p2q)/( (E + r**2 ) ** 1.5 * p1m) # use newton's law of universal gravitation, and coulomb's law (subtraction because opposites attract, like charges repel), divide by mass because of newton's 2nd law

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
    getForce = np.frompyfunc(getForceNV,16,6) # vectorize the function

######## MAIN PROGRAM FUNCTION ########
def main():
    global px, py, pz, pvx, pvy, pvz, pq, pm # global variables
    for n in range(iterations):

        # print out status
        if (n/iterations)*100 % 1 == 0 and n != 0 and OUTPUT == 3:
            now = round(time.time()-start_time,3)
            left = round(now*iterations/n-now,3)
            print((n/iterations)*100,"% complete\tETA:",str(left)+"s remaining ("+str(now)+"s elapsed)")

        # add frame to plot/video
        if n % frequency == 0:
            end_process.append([n, px.tolist(), py.tolist(), pz.tolist()])

        # temporary velocities
        tmp_vx, tmp_vy, tmp_vz = pvx, pvy, pvz

        if FRAMEWORK == "PyOpenCL-CPU" or FRAMEWORK == "PyOpenCL-GPU" or FRAMEWORK == "PyOpenCL":
            # transfer data to OpenCL
            px_g =   cl_array.to_device(queue, px)
            py_g =   cl_array.to_device(queue, py)
            pz_g =   cl_array.to_device(queue, pz)
            pvxt_g =   cl_array.to_device(queue, tmp_vx)
            pvyt_g =   cl_array.to_device(queue, tmp_vy)
            pvzt_g =   cl_array.to_device(queue, tmp_vz)
            pm_g =   cl_array.to_device(queue, pm)
            pq_g =   cl_array.to_device(queue, pq)
            
        if FRAMEWORK == "NumPy-M":
            for cp in numba.prange(p): # calculate forces on each particle
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
                        
                if FRAMEWORK == "PyOpenCL-CPU" or FRAMEWORK == "PyOpenCL-GPU" or FRAMEWORK == "PyOpenCL":
                    # buffers for retrieving gpu data
                    chg_vxg =  cl_array.empty_like(px_g)
                    chg_vyg =  cl_array.empty_like(py_g)
                    chg_vzg =  cl_array.empty_like(pz_g)
                    cls_vxg =  cl_array.empty_like(px_g)
                    cls_vyg =  cl_array.empty_like(py_g)
                    cls_vzg =  cl_array.empty_like(pz_g)
                    cp_g = cl_array.to_device(queue,np.array([cp]));

                    # calculate acceleration
                    prg.force(queue, px.shape, None,
                              cp_g.data, px_g.data, py_g.data, pz_g.data, pm_g.data, pq_g.data, chg_vxg.data, chg_vyg.data, chg_vzg.data, pvxt_g.data, pvyt_g.data, pvzt_g.data, cls_vxg.data, cls_vyg.data, cls_vzg.data).wait()

                    # copy data to cpu
                    chg_vx = chg_vxg.get()
                    chg_vy = chg_vyg.get()
                    chg_vz = chg_vzg.get()
                    cls_vx = cls_vxg.get()
                    cls_vy = cls_vyg.get()
                    cls_vz = cls_vzg.get()
                    
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
    if OUTPUT == 3:
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
        # array for coordinates, particles, and frames
        data_x = []
        data_y = []
        data_z = []
        data_p = []
        data_f = []

        # add numpy data to arrays
        for frame in frames:
            for p in range(len(frame[1])):
                data_x.append(frame[1][p])
                data_y.append(frame[2][p])
                data_z.append(frame[3][p])
                data_p.append(p)
                data_f.append(frame[0])

        # create data frame and scatter plot, then display in web browser
        data = pandas.DataFrame(data={'x':data_x,'y':data_y,'z':data_z,'f':data_f,'p':data_p})
        fig = plotlyx.scatter_3d(data, x='x', y='y', z='z', animation_frame='f', animation_group='p')
        fig.update_layout(scene=dict(xaxis=dict(range=[min(data_x), max(data_x)],autorange=False),yaxis=dict(range=[min(data_y), max(data_y)],autorange=False),zaxis=dict(range=[min(data_z), max(data_z)],autorange=False)))
        fig.show()

###### COMPILE #####
if OUTPUT >= 2:
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

if OUTPUT >= 2:
    print("Beginning N-body simulation")
start_time = time.time()    # start of program
main()                      # run program
midpoint_time = time.time() # runtime of program, exluding animation
create_video(end_process)   # create animation
end_time = time.time()      # runtime of program, including animation

if OUTPUT >= 2:
    print("Program has completed running using",FRAMEWORK)
if FRAMEWORK == "CuPy" and OUTPUT >= 2: print("Blocks/Threads:",num_blocks,"x",num_threads)
if OUTPUT >= 2:
    print(p,"particles for",iterations,"frames, recording every",frequency,"frames")
    print("Time to compile functions:     ",(((start_time-precompile_time)*100)//1)/100," seconds")
    print("Time to run N-body simulation: ",(((midpoint_time-start_time)*100)//1)/100," seconds")
    print("Time to create animation:      ",(((end_time-midpoint_time)*100)//1)/100," seconds")
    print("Total time:                    ",(((end_time-precompile_time)*100)//1)/100," seconds")
if OUTPUT == 1: print(midpoint_time-start_time)
