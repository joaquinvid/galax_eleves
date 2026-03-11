#ifdef GALAX_MODEL_GPU

#include <cmath>
#include <iostream>

#include "Model_GPU.hpp"
#include "kernel.cuh"

inline bool cuda_malloc(void ** devPtr, size_t size)
{
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(devPtr, size);
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "error: unable to allocate buffer" << std::endl;
        return false;
    }
    return true;
}

inline bool cuda_memcpy(void * dst, const void * src, size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t cudaStatus;
    cudaStatus = cudaMemcpy(dst, src, count, kind);
    if (cudaStatus != cudaSuccess)
    {
        std::cout << "error: unable to copy buffer" << std::endl;
        return false;
    }
    return true;
}

// UPDATE 1: Change signature to take float4* and drop massesGPU
void update_position_gpu(float4* pos_mass_GPU, float3* velocitiesGPU, float3* accelerationsGPU, int n_particles)
{
    update_position_cu(pos_mass_GPU, velocitiesGPU, accelerationsGPU, n_particles);
    cudaError_t cudaStatus;
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
        std::cout << "error: unable to synchronize threads" << std::endl;
}

Model_GPU::Model_GPU(const Initstate& initstate, Particles& particles)
: Model(initstate, particles),
  pos_mass_f4    (n_particles),
  velocitiesf3   (n_particles),
  accelerationsf3(n_particles)
{
    // init cuda
    cudaError_t cudaStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
        std::cout << "error: unable to setup cuda device" << std::endl;

    for (int i = 0; i < n_particles; i++)
    {
        velocitiesf3[i].x    = initstate.velocitiesx[i];
        velocitiesf3[i].y    = initstate.velocitiesy[i];
        velocitiesf3[i].z    = initstate.velocitiesz[i];

        pos_mass_f4[i].x = initstate.positionsx[i];
        pos_mass_f4[i].y = initstate.positionsy[i];
        pos_mass_f4[i].z = initstate.positionsz[i];
        pos_mass_f4[i].w = initstate.masses[i]; // Store mass in 'w'
    }

    // UPDATE 2: Allocate float4 and drop massesGPU
    cuda_malloc((void**)&pos_mass_GPU,     n_particles * sizeof(float4));
    cuda_malloc((void**)&velocitiesGPU,    n_particles * sizeof(float3));
    cuda_malloc((void**)&accelerationsGPU, n_particles * sizeof(float3));

    // UPDATE 3: Copy the combined float4 array and drop massesGPU copy
    cuda_memcpy(pos_mass_GPU,  pos_mass_f4.data(),  n_particles * sizeof(float4), cudaMemcpyHostToDevice);
    cuda_memcpy(velocitiesGPU, velocitiesf3.data(), n_particles * sizeof(float3), cudaMemcpyHostToDevice);
}

Model_GPU::~Model_GPU()
{
    // UPDATE 4: Clean up only what we allocated
    cudaFree(pos_mass_GPU);
    cudaFree(velocitiesGPU);
    cudaFree(accelerationsGPU);
}

void Model_GPU::step()
{
    cudaMemset(accelerationsGPU, 0, n_particles * sizeof(float3));

    // UPDATE 5: Pass the new float4 array
    update_position_gpu(pos_mass_GPU, velocitiesGPU, accelerationsGPU, n_particles);

    // UPDATE 6: Copy back to the float4 host array
    cuda_memcpy(pos_mass_f4.data(), pos_mass_GPU, n_particles * sizeof(float4), cudaMemcpyDeviceToHost);
    
    // UPDATE 7: Read coordinates back from the float4 array for the CPU renderer
    for (int i = 0; i < n_particles; i++)
    {
        particles.x[i] = pos_mass_f4[i].x;
        particles.y[i] = pos_mass_f4[i].y;
        particles.z[i] = pos_mass_f4[i].z;
        // No need to copy 'w' (mass) back since masses never change!
    }
}

#endif // GALAX_MODEL_GPU