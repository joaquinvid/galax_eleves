#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // CRITICAL: Prevent out-of-bounds memory access
    if (i >= n_particles) return;

    for (int j = 0; j < n_particles; j++)
    {
        if(i != j)
        {
            const float diffx = positionsGPU[j].x - positionsGPU[i].x;
            const float diffy = positionsGPU[j].y - positionsGPU[i].y;
            const float diffz = positionsGPU[j].z - positionsGPU[i].z;

            float dij = diffx * diffx + diffy * diffy + diffz * diffz;
            float multiplier = 0.0f;

            if (dij < 1.0f)
            {
                multiplier = 10.0f;
            }
            else
            {
                // Fast GPU math applied cleanly
                float inv_dist = rsqrtf(dij); 
                multiplier = 10.0f * (inv_dist * inv_dist * inv_dist);
            }

            // Using massesGPU[j] instead of initstate
            accelerationsGPU[i].x += diffx * multiplier * massesGPU[j];
            accelerationsGPU[i].y += diffy * multiplier * massesGPU[j];
            accelerationsGPU[i].z += diffz * multiplier * massesGPU[j];
        }
    }
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // CRITICAL: Prevent out-of-bounds memory access
    if (i >= n_particles) return;

    velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
    velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
    velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
    
    positionsGPU[i].x += velocitiesGPU[i].x * 0.1f;
    positionsGPU[i].y += velocitiesGPU[i].y * 0.1f;
    positionsGPU[i].z += velocitiesGPU[i].z * 0.1f;
}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
    int nthreads = 128;
    int nblocks =  (n_particles + (nthreads - 1)) / nthreads;

    compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
    maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}

#endif // GALAX_MODEL_GPU