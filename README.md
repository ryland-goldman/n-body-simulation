# Using CUDA to Increase the Accuracy and Performance of Particle-Particle *N*-Body Simulations

2022-23 Synopsys Research Project

[Ryland Goldman](https://www.rylandgoldman.com/) • Los Gatos High School

---

## Project Background

An *N*-body simulation is a simulated universe involving bodies either on a macro scale (i.e. planetary) or a micro scale (i.e. atomic). Bodies interact with each other with forces such as gravity. While most optimizations of *N*-body simulations involve particle-cluster interactions (such as the Barnes-Hut algorithm), this project is unique because it focuses on the more accurate direct particle-particle interactions. As such calculations take lots of CPU time, this project will be using parallelization with a graphics processing unit (GPU). In addition, accuracy can be improved on an atomic and astronomical scale by incorporating electromagnetic forces in addition to gravitational ones.

My goal is to create a more efficient and accurate method for an *N*-body simulation using CUDA with a GPU.

The control in this experiment is the processing time of an Intel CPU (12600K model), and the experimental group will be an Nvidia GPU (3090 model). The dependent variable is the amount of time it takes to complete each step in the simulation. The independent variables are the number of threads used (10,496 with GPU and 16 with CPU) and the number of particles simulated.

This experiment is mostly limited in two ways: first, since computer components are expensive, I only have one. I will not be able to test multi-GPU or multi-CPU setups. I’m also limited in the amount of random access memory (RAM). The computer I will be using has 24 GB of Video RAM for the GPU and 32 GB which can be allocated between the CPU and GPU, minus the RAM used for system operation. Less RAM means a smaller number of particles can be used.

### Flowchart

![Flowchart](https://www.rylandgoldman.com/files/asr/Flowchart.png)

### Example Output

https://user-images.githubusercontent.com/48637662/206875568-a330c5bc-1c59-4f3f-a3a8-1acc370b57a1.mp4

## Daily Progress and Goals

### 26 December 2022

**Goals**: 1) Implement the `CuPy` kernel for force and acceleration calculations.

The kernel now works and is providing output to the CPU. End users can choose to set the global `FRAMEWORK` variable to either "CuPy" or "NumPy", depending on how the simulation should be run. "NumPy" keeps it as it has been before, meanwhile "CuPy" runs the previously defined CUDA kernel. I also made some minor modifications to the output of the program, such as using the `agg` backend of `matplotlib` instead of `TkAgg` for potentially better performance. See commit [e124b5a](https://github.com/ryland-goldman/n-body-simulation/commit/e124b5a2df768d225ab771d698338cd0b223b21b).

### 22 December 2022

**Goals**: 1) Begin programming CUDA kernel

Using the `CuPy.RawKernel` function, I created a CUDA kernel called `get_force` which is written using C++. The kernel, once compiled, will act in the same way as the `getForce` function (which uses the `NumPy` library). First, the function computes the process ID using the thread and block numbers (stored in the `tid` variable). Then, the constants are redefined (as the Python code is not executed on GPU subunits). The acceleration components are calculated in an identical manner to `getForce` and transmitted back to the CPU via a pointer. See commit [f239416](https://github.com/ryland-goldman/n-body-simulation/commit/f239416c174eb63271585a24e28c4d18ccce7b36).

### 16 December 2022

**Goals**: 1) Fix slow performance of `NumPy` simulation

I optimized the computation of forces in two ways. First, I switched from using `np.vectorize` to using `np.frompyfunc`. The two are mostly equivalent, except for a couple features which are exclusive to `vectorize` (see [here](https://stackoverflow.com/a/11157577)). This brought a modest performance improvement, but for a more significant one, I merged the `getForce`, `xcomp`, `ycomp`, and `zcomp` functions all into one. Therefore, the amount of loops was cut in four, and less time was spent on recomputing data which had already been processed before (such as the angles). See commit [09904d7](https://github.com/ryland-goldman/n-body-simulation/commit/09904d7e0aed993361b9fcc00383071fa0b1b0cf).

### 14 December 2022

**Goals**: 1) Test performance of `NumPy` simulation

I modified the end of the program to store three timestamps, the start of the run, the point when the simulation ends, and the point when processing ends. Then, the runtime is printed out to the Python CLI. I ran five trials with each program, setting the initial conditions to the following:

- `Iterations = 1e4`
- `Frequency = 5e2`
- `Particles = 10`
- `Time constant = 1e-7`

From the five trials performed, the performance with NumPy was 28.566±0.47 seconds, and the performance without was 1.126±0.005 seconds. The error bars represent twice the standard error of the mean, and do not overlap, so the data is significant. Unexpectedly, the NumPy library slowed down the performance a lot. Something in the new program is not functioning correctly, and requires debugging before moving on to adding the CuPy library which supports CUDA. See commit [242a687](https://github.com/ryland-goldman/n-body-simulation/commit/242a687afbf6c9f623e251657cd14da2c9a00862).

![Screenshot 2022-12-14 140828](https://user-images.githubusercontent.com/48637662/207725687-3363cdd6-52df-4d41-8bd0-3ec25f5fd53b.png)

### 12 December 2022

**Goals**: 1) Add animation to program.

Today, I defined the `create_video` function, which takes an input of a list of frames, where each frame is a second list of coordinates, and uses `Matplotlib` to save a 3D graph to `C:\Nbody\files`. The frames are all strung together using `FFmpeg`, and the frame rate is determined by the ratio of iterations to frequency. The images are all removed afterwards, and the video is stored at `C:\Nbody\video.mp4`. See commit [a56a6fe](https://github.com/ryland-goldman/n-body-simulation/commit/a56a6fe0974d957fbc394b8b396b4afc32cd7f70).

### 10 December 2022

**Goals**: 1) Rewrite program using NumPy library

I worked for a few hours this morning to try and make some progress with NumPy. The original `ParticleIntr` function has now been broken up into `getForce`  (to calculate the gross acceleration) and three other functions (to convert it into x, y, and z components). All four are vectorized. The data is stored in eight separate arrays, due to the way `np.vectorize()` seems to work. The end result appears to parallel the original simulation. The next step is to store the results for the video animation. See commit [e795856](https://github.com/ryland-goldman/n-body-simulation/commit/e7958561f316bc9bd9fa125b10272ed93b961124).

### 8 December 2022

**Goals**: 1) Rewrite program using NumPy library

I'm continuing to convert the program to NumPy. Today, I worked to define the `ParticleIntrNV` function, which is converted to a vectorized function `ParticleIntr` using `np.vectorize()`. The `ParticleIntr` function takes an input of two particles, and calculates the forces between them using the distance formula, Newton's Law of Gravitation, Coulomb's Law, and trigonometry. See commit [bc5e20b](https://github.com/ryland-goldman/n-body-simulation/commit/bc5e20b362caf26f45be8b9451590a631835d4fd).

---

### 6 December 2022

**Goals**: 1) Rewrite program using NumPy library

I'm beginning to write the program using the NumPy library. Because `NumPy.vectorize` and `NumPy.frompyfuunc` require simple mathematical operations, it will make it easier to be converted into a CUDA kernel or OpenCL kernel later. I'm keeping the constants the same as before, and storing particles in a 2D array, with the second dimension storing the eight datapoints of the particles. See commit [ab0edfa](https://github.com/ryland-goldman/n-body-simulation/commit/ab0edfac793db095024de6adbf1ab36bc50d4880).

### 2 December 2022

**Goals**: 1) Review NumPy tutorials to rewrite program using NumPy library, 2) fill out Friday reflection for 2/12

No changes to the program today, but I worked to learn more about the Python `NumPy` library, using these sources: [1](https://www.w3schools.com/python/numpy/numpy_ufunc_create_function.asp), [2](https://stackoverflow.com/questions/6768245/difference-between-frompyfunc-and-vectorize-in-numpy), [3](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.frompyfunc.html#numpy.frompyfunc), [4](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html), [5](https://stackoverflow.com/questions/22981845/3-dimensional-array-in-numpy)

### 30 November 2022

**Goals**: 1) Set up PC and install libraries

I installed `Python 3.10.7`, `CuPy 11.3.0`, `Nvidia CUDA Toolkit 11.8`, `FFmpeg`, and `Matplotlib 3.6.2`, plus all necessary dependencies, onto my computer. I updated the code to run on a PC, for example by changing the shell script to parse the data with `Matplotlib`. The directories also needed to be updated from `/Users/rylandgoldman/...` to `C:\Nbody\...`, because of the Windows filesystem. See commit [4f71716](https://github.com/ryland-goldman/n-body-simulation/commit/4f71716a726795549b7a20c5313d2a4da88dc988).

### 28 November 2022

**Goals**: 1) Start status update presentation, 2) discuss notebook problems

I did not work on the project today. Instead, I worked on the presentation for my [4th status update](https://docs.google.com/presentation/d/1wuFTqlnsiTqC-TL3KJPcNL920kxUsa2ucYBVUHMBNU4/edit). I also discussed changes I could make to my [notebook](https://docs.google.com/document/d/1xeX6B97Fp9gwmoVqsXZ3Tb4dAV37uFZeaSvDWccRUDs/edit#) for the next notebook check on 2022-12-06.

---

### 21 November 2022

**Goals**: 1) Work more on collisions, 2) Measure performance

Today, I implemented support for collisions to the *N*-body simulation. The formula I used was `p1.vx = (p1vx*(p1.m-p2.m)+2*p2.m*p2vx)/(p1.m+p2.m)`. After calculating the collision, it pushes the particles apart until they are out of bounds of each other. I also changed the calculation of angle `alpha` in case `r` is sufficiently close to zero to cause a `ValueError`. See commit [fc172b5](https://github.com/ryland-goldman/n-body-simulation/commit/fc172b563e47ef51fa328915a714cb578c263323).

I also tested the performance of a simulation with `1e5` iterations, `1e3` frequency, and 20 randomly-generated particles. The compute times were 68.468 seconds, 67.083 seconds, 68.791 seconds, 66.789 seconds, and 66.561 seconds, with a standard deviation of 1.019 and standard error of the mean of 0.456.

### 17 November 2022

**Goals**: 1) Fill out Friday Reflection for 17/11, 2) complete journal write up #4

Today, I worked on my [Journal Write Up 4](https://docs.google.com/document/d/1SawG_NuJS8U4WbnQP5wHD6LZce_LIKmVna33sIwBS9w/edit#) and filled out this week's Friday Reflection.
