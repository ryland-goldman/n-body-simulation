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


import cupy

force_kernel = cupy.RawKernel(
    r'''
    extern "C" __global__
    void force_kernel(
        const double p1x, const double p1y, const double p1z, const double p1m, const double p1q, const double* p2x, const double* p2y, const double* p2z, const double* p2m, const double* p2q, double* p1vx, double* p1vy, double* p1vz
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;

        double G = 1;
        double k = 1;
        double E = 0.0000000000000000000000000000000001;
        double t = 0.01;

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

resultvx = cupy.zeros((100000), dtype='float64')
resultvy = cupy.zeros((100000), dtype='float64')
resultvz = cupy.zeros((100000), dtype='float64')
px = cupy.random.random((100000), dtype='float64');
py = cupy.random.random((100000), dtype='float64');
pz = cupy.random.random((100000), dtype='float64');
pm = cupy.random.random((100000), dtype='float64');
pq = cupy.random.random((100000), dtype='float64');

force_kernel((1000,),(100,),(1.0, 1.0, 1.0, 1.0, 1.0, px, py, pz, pm, pq, resultvx, resultvy, resultvz))
