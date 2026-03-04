#ifdef GALAX_MODEL_GPU

#include "cuda.h"
#include "kernel.cuh"
#define DIFF_T (0.1f)
#define EPS (1.0f)

__global__ void compute_acc(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, float* massesGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	for (int i = 0; i < n_particles; i ++)
    {
        for (int j = 0; j < n_particles; j++)
        {
            if(i != j)
            {
                const float diffx = particles.x[j] - particles.x[i];
                const float diffy = particles.y[j] - particles.y[i];
                const float diffz = particles.z[j] - particles.z[i];

                float dij = diffx * diffx + diffy * diffy + diffz * diffz;

                if (dij < 1.0)
                {
                    dij = 10.0;
                }
                else
                {
                    dij = std::sqrt(dij);
                    dij = 10.0 / (dij * dij * dij);
                }

                accelerationsx[i] += diffx * dij * initstate.masses[j];
                accelerationsy[i] += diffy * dij * initstate.masses[j];
                accelerationsz[i] += diffz * dij * initstate.masses[j];
            }
        }
    }
}

__global__ void maj_pos(float3 * positionsGPU, float3 * velocitiesGPU, float3 * accelerationsGPU, int n_particles)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < n_particles; i++)
	{
		velocitiesx[i] += accelerationsx[i] * 2.0f;
		velocitiesy[i] += accelerationsy[i] * 2.0f;
		velocitiesz[i] += accelerationsz[i] * 2.0f;
		particles.x[i] += velocitiesx   [i] * 0.1f;
		particles.y[i] += velocitiesy   [i] * 0.1f;
		particles.z[i] += velocitiesz   [i] * 0.1f;
	}

}

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles)
{
	int nthreads = 128;
	int nblocks =  (n_particles + (nthreads -1)) / nthreads;

	compute_acc<<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, massesGPU, n_particles);
	maj_pos    <<<nblocks, nthreads>>>(positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}


#endif // GALAX_MODEL_GPU
