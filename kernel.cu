#include <cuda_runtime.h>
        extern "C" __global__

        /*
         * For comments, please see the OpenCL implementation.
         */
        void force_kernel(
            double p1x, double p1y, double p1z, double p1vxi, double p1vyi, double p1vzi, double p1m, double p1q, double* p2x, double* p2y, double* p2z, double* p2m, double* p2q, double* p1vx, double* p1vy, double* p1vz, double* p2vx, double* p2vy, double* p2vz, double* v1x, double* v1y, double* v1z
        ) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            
            //ImportConstants

            double dx = p1x - p2x[tid];
            double dy = p1y - p2y[tid];
            double dz = p1z - p2z[tid];
            
            double r = sqrt( dx*dx + dy*dy + dz*dz );
            if( r != 0.0 ){
                double f = t * r * (G * p1m * p2m[tid] - k * p1q * p2q[tid])/(math.sqrt((r * r + E)*(r * r + E)*(r * r + E))*p1m);

                p1vx[tid] = -1.0 * f * dx / r;
                p1vy[tid] = -1.0 * f * dy / r;
                p1vz[tid] = -1.0 * f * dz / r;
                
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
        }
