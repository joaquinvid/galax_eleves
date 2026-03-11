#ifdef GALAX_MODEL_GPU

#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include <stdio.h>

//void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles);

// UPDATED: Now takes float4* and drops the float* massesGPU
void update_position_cu(float4* pos_mass_GPU, float3* velocitiesGPU, float3* accelerationsGPU, int n_particles);

#endif

#endif // GALAX_MODEL_GPU
