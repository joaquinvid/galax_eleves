#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)
#define BLOCK_SIZE 256 

// we now pass float4* for positions (which holds x,y,z,mass)
__global__ void compute_acc(float4 * pos_mass_GPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tx = threadIdx.x;
    
    float acc_x = 0.0f;
    float acc_y = 0.0f;
    float acc_z = 0.0f;

    // We use a float4 to hold our own position AND mass
    float4 my_pos_mass = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (i < n_particles) {
        my_pos_mass = pos_mass_GPU[i]; 
    }

    // SHARED MEMORY: Only ONE array needed now, perfectly aligned!
    __shared__ float4 sh_pos_mass[BLOCK_SIZE];

    int num_tiles = (n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < num_tiles; tile++)
    {
        int load_idx = tile * BLOCK_SIZE + tx;
        
        // 128-bit Vector Load! Fast and coalesced.
        if (load_idx < n_particles) {
            sh_pos_mass[tx] = pos_mass_GPU[load_idx];
        } else {
            sh_pos_mass[tx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f); // Pad with zero mass
        }
        
        __syncthreads();

        if (i < n_particles) 
        {
            #pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j++) 
            {
                // Read position and mass from shared memory
                float4 p_j = sh_pos_mass[j]; 

                float diffx = p_j.x - my_pos_mass.x;
                float diffy = p_j.y - my_pos_mass.y;
                float diffz = p_j.z - my_pos_mass.z;

                float dij = diffx * diffx + diffy * diffy + diffz * diffz;
                
                // branch eliminated
                // 1. Calculate mult_far for the 'else' case
                float inv_dist = rsqrtf(dij); 
                float mult_far = 10.0f * (inv_dist * inv_dist * inv_dist);
                
                // 2. Use a ternary operator to select the multiplier. 
                // The compiler turns this into a fast "SEL" instruction (zero divergence)
                float multiplier = (dij < 1.0f) ? 10.0f : mult_far;

                // 3. Add to accumulators. (p_j.w is the mass!)
                // If i == j, diffx is 0, so it naturally adds 0.0f. No 'if(i!=j)' needed!
                acc_x += diffx * multiplier * p_j.w;
                acc_y += diffy * multiplier * p_j.w;
                acc_z += diffz * multiplier * p_j.w;
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

__global__ void maj_pos(float4 * pos_mass_GPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n_particles) return;

    velocitiesGPU[i].x += accelerationsGPU[i].x * 2.0f;
    velocitiesGPU[i].y += accelerationsGPU[i].y * 2.0f;
    velocitiesGPU[i].z += accelerationsGPU[i].z * 2.0f;
    
    // Update the x, y, and z components of the float4
    pos_mass_GPU[i].x += velocitiesGPU[i].x * 0.1f;
    pos_mass_GPU[i].y += velocitiesGPU[i].y * 0.1f;
    pos_mass_GPU[i].z += velocitiesGPU[i].z * 0.1f;
}

void update_position_cu(float4* pos_mass_GPU, float3* velocitiesGPU, float3* accelerationsGPU, int n_particles)
{
    int nblocks =  (n_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    compute_acc<<<nblocks, BLOCK_SIZE>>>(pos_mass_GPU, velocitiesGPU, accelerationsGPU, n_particles);
    maj_pos    <<<nblocks, BLOCK_SIZE>>>(pos_mass_GPU, velocitiesGPU, accelerationsGPU, n_particles);
}

#endif // GALAX_MODEL_GPU