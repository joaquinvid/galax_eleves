#ifdef GALAX_MODEL_CPU_FAST
#include <cmath>
#include <algorithm>
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
    const b_type logic_threshold(1.0f);
    const b_type logic_default(10.0f);
    const int N = n_particles;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i += b_type::size)
    {
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        
        b_type raccx_i(0.0f);
        b_type raccy_i(0.0f);
        b_type raccz_i(0.0f);

        // Inner loop: process in blocks of 8 to match SIMD width
        for (int j = 0; j < N; j += b_type::size)
        {
            // Instead of loading a batch and extracting, we interact 
            // the i-batch with each of the 8 particles in the j-block.
            // This ensures we stay within the L1 cache.
            for (int k = 0; k < b_type::size && (j + k) < N; ++k)
            {
                // Broadcast scalars to b_type
                const b_type pkx(particles.x[j + k]);
                const b_type pky(particles.y[j + k]);
                const b_type pkz(particles.z[j + k]);
                const b_type mk(initstate.masses[j + k]);

                const b_type diffx = pkx - rposx_i;
                const b_type diffy = pky - rposy_i;
                const b_type diffz = pkz - rposz_i;

                b_type dij_sq = (diffx * diffx) + (diffy * diffy) + (diffz * diffz);

                auto mask = dij_sq < logic_threshold;
                b_type inv_dist = xs::rsqrt(dij_sq);
                b_type inv_dist3 = inv_dist * inv_dist * inv_dist;
                
                b_type final_dij = xs::select(mask, logic_default, logic_default * inv_dist3);

                raccx_i += diffx * final_dij * mk;
                raccy_i += diffy * final_dij * mk;
                raccz_i += diffz * final_dij * mk;
            }
        }

        raccx_i.store_unaligned(&accelerationsx[i]);
        raccy_i.store_unaligned(&accelerationsy[i]);
        raccz_i.store_unaligned(&accelerationsz[i]);
    }

    // 2. Integration Loop
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i += b_type::size)
    {
        b_type vx = b_type::load_unaligned(&velocitiesx[i]);
        b_type vy = b_type::load_unaligned(&velocitiesy[i]);
        b_type vz = b_type::load_unaligned(&velocitiesz[i]);
        
        const b_type ax = b_type::load_unaligned(&accelerationsx[i]);
        const b_type ay = b_type::load_unaligned(&accelerationsy[i]);
        const b_type az = b_type::load_unaligned(&accelerationsz[i]);

        vx += ax * 2.0f;
        vy += ay * 2.0f;
        vz += az * 2.0f;

        vx.store_unaligned(&velocitiesx[i]);
        vy.store_unaligned(&velocitiesy[i]);
        vz.store_unaligned(&velocitiesz[i]);

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