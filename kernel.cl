
// Enable OpenCL extension for double-precision FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void force(
        __global const int *cp_ptr,   // Pointer for the integer index of the current particle in the array
        __global const double *p2x,   // X-coordinate array of all particles
        __global const double *p2y,   // Y-coordinate array of all particles
        __global const double *p2z,   // Z-coordinate array of all particles
        __global const double *p2m,   // Mass of all particles
        __global const double *p2q,   // Charge of all particles
        __global double *p1vx,        // Modifiable array of partial component velocities in the x direction
        __global double *p1vy,        // Modifiable array of partial component velocities in the y direction
        __global double *p1vz,        // Modifiable array of partial component velocities in the z direction
        __global const double *p2vx,  // X component direction of all particle velocities
        __global const double *p2vy,  // Y component direction of all particle velocities
        __global const double *p2vz,  // Z component direction of all particle velocities
        __global double *v1x,         // Contains new velocitiy in the x direction if a collision occured, otherwise zero
        __global double *v1y,         // Contains new velocitiy in the x direction if a collision occured, otherwise zero
        __global double *v1z          // Contains new velocitiy in the x direction if a collision occured, otherwise zero
    ){
        // Fetch current particle information using data from cp_ptr
        int cp = cp_ptr[0];
        double p1x = p2x[cp];
        double p1y = p2y[cp];
        double p1z = p2z[cp];
        double p1vxi = p2vx[cp];
        double p1vyi = p2vy[cp];
        double p1vzi = p2vz[cp];
        double p1m = p2m[cp];
        double p1q = p2q[cp];

        // Get ID of current thread
        int tid = get_global_id(0);

        //ImportConstants
        

        // Calculate differences in position
        double dx = p1x - p2x[tid];
        double dy = p1y - p2y[tid];
        double dz = p1z - p2z[tid];

        // Calculate total distance with distance formula
        double r = sqrt( dx*dx + dy*dy + dz*dz );

        // Check if particles are different. If so, continue. If not, skip the particle.
        if( r != 0.0 ){
            double f_g = G * p1m * p2m[tid];  // Force from gravity
            double f_e = k * p1q * p2q[tid];  // Force from electromagnetsim
            double f = t * r * (f_g - f_e)/( sqrt((r * r + E)*(r * r + E)*(r * r + E)) * p1m);  // Net acceleration 

            // Calculate acceleration components
            p1vx[tid] = -1.0 * f * dx / r;
            p1vy[tid] = -1.0 * f * dy / r;
            p1vz[tid] = -1.0 * f * dz / r;

            // If a collision occured, set the new velocities assuming perfectly elastic collisions
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
