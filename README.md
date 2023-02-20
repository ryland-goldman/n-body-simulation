# Using CUDA to Increase the Accuracy and Performance of Particle-Particle *N*-Body Simulations

2022-23 Synopsys Research Project

Project Number 112-H10-31

[Ryland Goldman](https://www.rylandgoldman.com/) • Los Gatos High School

---

## Abstract

*N*-body simulations are computer models of systems of bodies typically used for physics and astronomy applications such as predicting orbits of planets. These simulations typically require large amounts of processing power and time for physics computations. To solve this issue, developers use rounding and make compromises on accuracy in order to optimize the software. This project aims to use hardware acceleration rather than mathematical approximations to improve the performance of the simulation, written in Python.

The project compares a NumPy-based approach running on a 16-thread Intel 12600K CPU (compiled with Numba JIT) with CuPy interfacing with a NVIDIA 3090 GPU via the CUDA framework. The CPU group was the control, and CUDA was the experimental group. Two additional test groups used PyOpenCL to directly compare each device. One hundred trials were run on each of the four groups, and repeated using powers of two between 2^13 and 2^18 bodies.

Using 2^16 bodies, the speed up multiple for CuPy was 3.66x, OpenCL (GPU) was 1.05x, and OpenCL (CPU) was 0.56x. This suggests that CUDA is significantly faster than only using the CPU for computations, and the GPU OpenCL implementation was about twice as fast as the CPU OpenCL implementation.

### Flowchart

![Flowchart](https://www.rylandgoldman.com/files/asr/Flowchart.png)

### Example Output

https://user-images.githubusercontent.com/48637662/218208079-0eaac62a-7e47-48f6-a5a3-dbfc718e27f6.mov

## Dependencies

- Windows 11 Home Build 22621.900
- Python 3.10.7 with pip CLI
- Python random, time, os, sys, and math builtin libraries
- Nvidia CUDA Toolkit 11.8
- CuPy 11.3.0
- NumPy 1.23.3
- Fastrlock 0.8.1
- Matplotlib 3.6.2
- Contourpy 1.0.6
- Cycler 0.11.0
- Fonttools 4.38.0
- Kiwisolver 1.4.4
- Pillow 9.3.0
- Dateutil 2.8.2
- Six 1.16.0
- Pyparsing 3.0.9
- Packaging 20.0
- FFmpeg 5.1.2
- PyOpenCL 2022.3.1
- Pytools 2022.1.14
- Platformdirs 2.6.2
- Typing-extensions 4.4.0
- Plotly 5.12.0
- Tenacity 8.1.0
- Pandas 1.5.3
- Pytz 2022.7.1
- Numba 0.56.4
- Setuptools 65.6.3
- Llvmlite 0.39.1

## Data

![Scatter](https://user-images.githubusercontent.com/48637662/219972651-b9b71811-944d-45ee-a607-eefff1c98a7d.png)
![Bar](https://user-images.githubusercontent.com/48637662/219972653-302bc70a-4dc3-48c2-967f-f5c17ef2dd3a.png)


| Number of Bodies | Data File |
| ---------------- | --------- |
| 8192 | [DATA-8192.txt](https://rgold.dev/files/asr/DATA-8192.txt) |
| 16384 | [DATA-16384.txt](https://rgold.dev/files/asr/DATA-16384.txt) |
| 32768 | [DATA-32768.txt](https://rgold.dev/files/asr/DATA-32768.txt) |
| 65536 | [DATA-65536.txt](https://rgold.dev/files/asr/DATA-65536.txt) |
| 131072 | [DATA-131072.txt](https://rgold.dev/files/asr/DATA-131072.txt) |
| 262144 | [DATA-262144.txt](https://rgold.dev/files/asr/DATA-262144.txt) |

### Conclusion

OpenCL seems consistently slower than native code (NumPy on CPU, CUDA on GPU). Interestingly, the OpenCL CPU was faster than the NumPy CPU when p ≥ 65,536. At this point, it was approximately the same as the OpenCL GPU, suggesting that the speed might have been limited by the latency of random access memory.

The CUDA code has a trendline (ax^b) where b=0.978, meaning that the growth is less than linear. The most likely cause is the large relative inter-device data transfer times when the number of bodies is small, which have less of an effect when the compute time is greater. This also explains why NumPy is the fastest framework when p=8,192.

OpenCL on the CPU was the most consistent test group. The R^2 of the trendline was 0.999. This is likely due to the fact that it is compiled in C++ instead of being interpreted with Python or sent to other devices.

When the number of bodies was 16,384 ≤ p ≤ 65,536, some of the test groups were bimodal (as noted in the higher standard errors). The cause might be that some test groups had a higher frequency of collisions which took longer to calculate (since the initial positions and velocities were randomly chosen). The 8,192 group didn’t have many collisions since there were fewer particles, while the 131,072 and 262,144 groups had enough collisions that every trial encountered at least one.

## Daily Progress and Goals

### 8 February 2023 to 20 February 2023

**Goals**: 1) Analyse data, 2) clean up code

For the code used for the data analysis, please see [Analysis](https://github.com/ryland-goldman/n-body-simulation/tree/main/analysis). I also cleaned up the code for better presentation, moving the kernels to seperate files, and added a small fix to force softening.

### 23 January 2023 to 6 February 2023

**Goals**: 1) Collect data.

Using a separate Python script, I’m looping through each of the four frameworks and testing one hundred trials of each, averaging the data, and sending it to my web server. The script will be run six times, once on each group (2^13, 2^14, 2^15, 2^16, 2^17, and 2^18 bodies). The parameters of the test script force one iteration, no rendering, and only a `float` timestamp output. Constants are set to `G=3,000`, `k=0`, `E=2^-128`, `t=0.01`, and `s=0.05`. Particle locations/positions are randomly generated, while mass/charge is set to one.

### 17 January 2023 to 19 February 2023

**Goals**: 1) Run preliminary performance benchmark

Run the program with each starting condition (4,096, 8,192, 16,384, 32,768, 65,536, and 131,072 bodies) once on each device. The output is discarded (as I only need the program runtime). Constants are set to `G=3,000`, `k=0`, `E=2^-128`, `t=0.01`, and `s=0.05`. Particle locations/positions are randomly generated, while mass/charge is set to one.

Blank cells took >600 seconds
| Number of Particles | GPU | CPU Multithread | CPU Single Thread |
| ------- | ------- | ------- | ------- |
| 4096 | 1.47 | 0.57 | 48.66 |
| 8192 | 2.98 | 1.79 | 199.36 |
| 16384 | 11.83 | 5.86 | |
| 32768 | 21.35 | 21.16 | |
| 65536 | 50.17 | 114.79 | |
| 131072 | 110.26 | | |

![Processing Time vs  Number of Particles](https://user-images.githubusercontent.com/48637662/218206137-c4e6c3aa-7193-431c-b3f4-2185b781c3dc.png)


This data has no statistical significance since only one trial was performed. However, the trendlines (ax^b) were around b=2 for most groups, as expected. The “CPU Multi” category ran as expected (exponential growth). The “GPU CUDA” category took longer than “CPU Multi” at the start, likely due to inter-device data transfer delays, eventually speeding up by 2^17 particles. The “CPU Single” took a lot longer than expected — instead of being 16x slower (as I have a 16-thread CPU), it was around 100x slower. I do not have time to use the CPU single thread in the final experiment.

### 13 January 2023

**Goals**: 1) Enable pre-compilation of the `NumPy` and `CuPy` functions.

The `NumPy` and `CuPy` functions need to be compiled before they are run. Since this is done using JIT (just-in-time) compilation, it happens at the moment they are first called. This slows down the simulation and can introduce another variable to the results. By calling the functions once before the timer starts, and therefore compiling them early, the variable can be eliminated. See commit [5a13b73](https://github.com/ryland-goldman/n-body-simulation/commit/5a13b730089e4f0db8e9a42eedfbcd50a3aebfe2).

### 9 January 2023

**Goals**: 1) Modify output of simulation to have an option to use `Plotly` library

The variable `DISPLAY` now has three states. "Video" keeps the current behaviour (plotting with `matplotlib` and combining frames with `FFmpeg`). "Plot" uses the `Plotly` library to create an interactive web-browser based plot (note, the plot is stored to RAM, not to disk). "Both" makes a video and a plot. It creates a data frame with the `Pandas` library that is used to form a 3D scatter plot. `Plotly` is significantly faster than `matplotlib`. See commit [82de489](https://github.com/ryland-goldman/n-body-simulation/commit/82de4890aeef0a6e7d9e3ec125967975e75ff2e2).

### 8 January 2023

**Goals**: 1) Add multithreading using `NumPy`, 2) clean up program code, 3) work on handling collisions

The code now has a better interface, including providing a progress bar. I also added multithreading for the `NumPy` (CPU) N-body simulation. This was done using just-in-time (JIT) compilation provided by the `Numba` library. I cleaned up some of the code to make it more readable. Lastly, I think I finally got collisions to work succesfully (if the distance is less than the value in the `s` variable; collisions are assumed to be completely elastic). See commit [7234bc4](https://github.com/ryland-goldman/n-body-simulation/commit/7234bc437218de0dcab76483cba8fd7f76142e37).

### 29 December 2022

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
