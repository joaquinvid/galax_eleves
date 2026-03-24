#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>
#include "Model_CPU_fast.hpp"
#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;
using b_type = xs::batch<float, xs::avx2>;

Model_CPU_fast::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast::step()
{
    // 1. Reset accelerations
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0.0f);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0.0f);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0.0f);


    // 2. Compute Accelerations (Parallel + Vectorized)
    const int n = n_particles;
    const int b_size = b_type::size;
    const int vec_limit = n - (n % b_size); // The last index divisible by batch size

    // 1. Vectorized Loop (Main Workload)
    #pragma omp parallel for
    for (int i = 0; i < vec_limit; i += b_size)
    {
        b_type px_i = b_type::load_unaligned(&particles.x[i]);
        b_type py_i = b_type::load_unaligned(&particles.y[i]);
        b_type pz_i = b_type::load_unaligned(&particles.z[i]);
        
        b_type acc_ix = 0.0f, acc_iy = 0.0f, acc_iz = 0.0f;

        for (int j = 0; j < n; j++)
        {
            b_type dx = particles.x[j] - px_i;
            b_type dy = particles.y[j] - py_i;
            b_type dz = particles.z[j] - pz_i;
            b_type r2 = dx * dx + dy * dy + dz * dz;

            auto mask = r2 < 1.0f;
            b_type dij = xs::select(mask, b_type(10.0f), 10.0f / (r2 * xs::sqrt(r2)));

            b_type factor = dij * initstate.masses[j];
            acc_ix += dx * factor;
            acc_iy += dy * factor;
            acc_iz += dz * factor;
        }

        acc_ix.store_unaligned(&accelerationsx[i]);
        acc_iy.store_unaligned(&accelerationsy[i]);
        acc_iz.store_unaligned(&accelerationsz[i]);
    }

    for (int i = vec_limit; i < n; i++)
    {
        float accx = 0, accy = 0, accz = 0;
        for (int j = 0; j < n; j++)
        {
            if (i == j) continue;
            float dx = particles.x[j] - particles.x[i];
            float dy = particles.y[j] - particles.y[i];
            float dz = particles.z[j] - particles.z[i];
            float r2 = dx * dx + dy * dy + dz * dz;

            float dij = (r2 < 1.0f) ? 10.0f : 10.0f / (std::sqrt(r2) * r2);
            
            accx += dx * dij * initstate.masses[j];
            accy += dy * dij * initstate.masses[j];
            accz += dz * dij * initstate.masses[j];
        }
        accelerationsx[i] = accx;
        accelerationsy[i] = accy;
        accelerationsz[i] = accz;
    }

    // 3. Integration Step (Update Velocity and Position)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_particles; i += b_type::size)
    {
        b_type vx = b_type::load_unaligned(&velocitiesx[i]);
        b_type vy = b_type::load_unaligned(&velocitiesy[i]);
        b_type vz = b_type::load_unaligned(&velocitiesz[i]);
        
        const b_type ax = b_type::load_unaligned(&accelerationsx[i]);
        const b_type ay = b_type::load_unaligned(&accelerationsy[i]);
        const b_type az = b_type::load_unaligned(&accelerationsz[i]);

        // velocities += accelerations * 2.0f
        vx += ax * 2.0f;
        vy += ay * 2.0f;
        vz += az * 2.0f;

        vx.store_unaligned(&velocitiesx[i]);
        vy.store_unaligned(&velocitiesy[i]);
        vz.store_unaligned(&velocitiesz[i]);

        // particles += velocities * 0.1f
        b_type px = b_type::load_unaligned(&particles.x[i]);
        b_type py = b_type::load_unaligned(&particles.y[i]);
        b_type pz = b_type::load_unaligned(&particles.z[i]);

        px += vx * 0.1f;
        py += vy * 0.1f;
        pz += vz * 0.1f;

        px.store_unaligned(&particles.x[i]);
        py.store_unaligned(&particles.y[i]);
        pz.store_unaligned(&particles.z[i]);
    }
}

#endif