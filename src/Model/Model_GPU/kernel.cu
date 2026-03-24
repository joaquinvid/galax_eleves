#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

#define BLOCK_SIZE 256 

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tx = threadIdx.x;
    
    float acc_x = 0.0f;
    float acc_y = 0.0f;
    float acc_z = 0.0f;

    // Cache this thread's particle position so we don't read it 10,000 times
    float3 my_pos = make_float3(0.0f, 0.0f, 0.0f);
    if (i < n_particles) {
        my_pos = positionsGPU[i];
    }

    // ALLOCATE SHARED MEMORY: This acts as an ultra-fast L1 cache for the block
    __shared__ float3 sh_pos[BLOCK_SIZE];
    __shared__ float  sh_mass[BLOCK_SIZE];

    // Calculate how many "tiles" (chunks of 256 particles) we need to process
    int num_tiles = (n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < num_tiles; tile++)
    {
        int load_idx = tile * BLOCK_SIZE + tx;
        if (load_idx < n_particles) {
            sh_pos[tx] = positionsGPU[load_idx];
            sh_mass[tx] = massesGPU[load_idx];
        } else {
            sh_mass[tx] = 0.0f; 
        }
        
        __syncthreads();

        if (i < n_particles) 
        {
            #pragma unroll // Tells the compiler to optimize this loop heavily
            for (int j = 0; j < BLOCK_SIZE; j++) 
            {
                int global_j = tile * BLOCK_SIZE + j;
                
                // Don't calculate gravity against ourselves, and don't read padded fake particles
                if (global_j < n_particles && i != global_j) 
                {
                    // Notice we read from sh_pos, NOT positionsGPU!
                    const float diffx = sh_pos[j].x - my_pos.x;
                    const float diffy = sh_pos[j].y - my_pos.y;
                    const float diffz = sh_pos[j].z - my_pos.z;

                    float dij = diffx * diffx + diffy * diffy + diffz * diffz;
                    float multiplier = 0.0f;

                    if (dij < 1.0f) {
                        multiplier = 10.0f;
                    } else {
                        float inv_dist = rsqrtf(dij); 
                        multiplier = 10.0f * (inv_dist * inv_dist * inv_dist);
                    }

                    // Add to local registers
                    acc_x += diffx * multiplier * sh_mass[j];
                    acc_y += diffy * multiplier * sh_mass[j];
                    acc_z += diffz * multiplier * sh_mass[j];
                }
            }
        }
        
        __syncthreads();
    }

    if (i < n_particles) {
        accelerationsGPU[i].x += acc_x;
        accelerationsGPU[i].y += acc_y;
        accelerationsGPU[i].z += acc_z;
    }
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
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
    int nblocks =  (n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    compute_acc<<<nblocks, BLOCK_SIZE>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
    maj_pos    <<<nblocks, BLOCK_SIZE>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}

#endif // GALAX_MODEL_GPU