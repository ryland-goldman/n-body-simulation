# Using CUDA to Increase the Accuracy and Performance of Particle-Particle N-Body Simulations

2022-23 Synopsys Research Project

[Ryland Goldman](https://www.rylandgoldman.com/) • Los Gatos High School

---

## Project Background

An N-body simulation is a simulated universe involving bodies either on a macro scale (i.e. planetary) or a micro scale (i.e. atomic). Bodies interact with each other with forces such as gravity. While most optimizations of N-body simulations involve particle-cluster interactions (such as the Barnes-Hut algorithm), this project is unique because it focuses on the more accurate direct particle-particle interactions. As such calculations take lots of CPU time, this project will be using parallelization with a graphics processing unit (GPU). In addition, accuracy can be improved on an atomic and astronomical scale by incorporating electromagnetic forces in addition to gravitational ones.

## Daily Goals

### 8 December 2022

I'm continuing to convert the program to NumPy. Today, I worked to define the `ParticleIntrNV` function, which is converted to a vectorized function `ParticleIntr` using `np.vectorize()`. The `ParticleIntr` function takes an input of two particles, and calculates the forces between them using the distance formula, Newton's Law of Gravitation, Coulomb's Law, and trigonometry. See commit [bc5e20b](https://github.com/ryland-goldman/n-body-simulation/commit/bc5e20b362caf26f45be8b9451590a631835d4fd).

### 6 December 2022

I'm beginning to write the program using the NumPy library. Because `NumPy.vectorize` and `NumPy.frompyfuunc` require simple mathematical operations, it will make it easier to be converted into a CUDA kernel or OpenCL kernel later. I'm keeping the constants the same as before, and storing particles in a 2D array, with the second dimension storing the eight datapoints of the particles. See commit [ab0edfa](https://github.com/ryland-goldman/n-body-simulation/commit/ab0edfac793db095024de6adbf1ab36bc50d4880).

### 2 December 2022

No changes to the program today, but I worked to learn more about the Python `NumPy` library, using these sources: [1](https://www.w3schools.com/python/numpy/numpy_ufunc_create_function.asp), [2](https://stackoverflow.com/questions/6768245/difference-between-frompyfunc-and-vectorize-in-numpy), [3](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.frompyfunc.html#numpy.frompyfunc), [4](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html), [5](https://stackoverflow.com/questions/22981845/3-dimensional-array-in-numpy)

### 30 November 2022

I installed `Python 3.10.7`, `CuPy 11.3.0`, `Nvidia CUDA Toolkit 11.8`, `FFmpeg`, and `Matplotlib 3.6.2`, plus all necessary dependencies, onto my computer. I updated the code to run on a PC, for example by changing the shell script to parse the data with `Matplotlib`. The directories also needed to be updated from `/Users/rylandgoldman/...` to `C:\Nbody\...`, because of the Windows filesystem. See commit [4f71716](https://github.com/ryland-goldman/n-body-simulation/commit/4f71716a726795549b7a20c5313d2a4da88dc988).

### 28 November 2022

I did not work on the project today. Instead, I worked on the presentation for my [4th status update](https://docs.google.com/presentation/d/1wuFTqlnsiTqC-TL3KJPcNL920kxUsa2ucYBVUHMBNU4/edit). I also discussed changes I could make to my [notebook](https://docs.google.com/document/d/1xeX6B97Fp9gwmoVqsXZ3Tb4dAV37uFZeaSvDWccRUDs/edit#) for the next notebook check on 2022-12-06.

### 21 November 2022

Today, I implemented support for collisions to the *N*-body simulation. The formula I used was `p1.vx = (p1vx*(p1.m-p2.m)+2*p2.m*p2vx)/(p1.m+p2.m)`. After calculating the collision, it pushes the particles apart until they are out of bounds of each other. I also changed the calculation of angle `alpha` in case `r` is sufficiently close to zero to cause a `ValueError`. See commit [fc172b5](https://github.com/ryland-goldman/n-body-simulation/commit/fc172b563e47ef51fa328915a714cb578c263323).

I also tested the performance of a simulation with `1e5` iterations, `1e3` frequency, and 20 randomly-generated particles. The compute times were 68.468 seconds, 67.083 seconds, 68.791 seconds, 66.789 seconds, and 66.561 seconds, with a standard deviation of 1.019 and standard error of the mean of 0.456.
