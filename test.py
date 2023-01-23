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

PRG_PATH = "C:\\Users\\rylan\\AppData\\Local\\Programs\\Python\\Python310\\asr.py"
import subprocess
import numpy as np
from math import sqrt

# the program accepts the following arguments:
# Framework - either "NumPy", "NumPy-M", "CuPy", "PyOpenCL-GPU", "PyOpenCL-CPU"
# Display   - either "Plot", "Video", or "None"
# Particles - number of particles in the simulation
# Iterations- number of iterations in the simulation

print("Please select a framework to test:")
print("==================================")
print("    1. NumPy (Single Thread)")
print("    2. NumPy (Multi Thread)")
print("    3. CUDA GPU")
print("    4. OpenCL CPU")
print("    5. OpenCL GPU")
FRAMEWORK = ["NumPy","NumPy-M","CuPy","PyOpenCL-CPU","PyOpenCL-GPU"][int(input("Enter your choice: "))-1]
print("")
PARTICLES = str(int(input("How many particles to use? ")))
print("")
TESTS = int(input("How many trials? "))
times = np.array([])
for n in range(TESTS):
    tbin = subprocess.check_output(["python",PRG_PATH,FRAMEWORK,"None",PARTICLES,"1"])
    t = float(str(tbin)[2:-5])
    times = np.append(times,t)
    print("Completed test",n+1,"of",TESTS,"in",t,"seconds")
print("Average time:",times.mean())
print("Standard deviation:",times.std())
se = times.std()/sqrt(TESTS)
print("Standard error:",se)
z = 1.9599639845400545 # 95% confidence
print("95% confidence interval: ("+str(times.mean() - z*se)+", "+str(times.mean() + z*se)+")")
