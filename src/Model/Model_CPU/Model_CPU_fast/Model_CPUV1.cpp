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

    // Constants for the logic
    const b_type logic_threshold(1.0f);
    const b_type logic_default(10.0f);

    // 2. Compute Accelerations (N*N interaction)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_particles; i += b_type::size)
    {
        const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
        const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
        const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
        
        b_type raccx_i(0.0f);
        b_type raccy_i(0.0f);
        b_type raccz_i(0.0f);

        for (int j = 0; j < n_particles; ++j)
        {
            const b_type posx_j(particles.x[j]);
            const b_type posy_j(particles.y[j]);
            const b_type posz_j(particles.z[j]);
            const b_type mass_j(initstate.masses[j]);

            const b_type diffx = posx_j - rposx_i;
            const b_type diffy = posy_j - rposy_i;
            const b_type diffz = posz_j - rposz_i;

            b_type dij_sq = (diffx * diffx) + (diffy * diffy) + (diffz * diffz);

            // SIMD Branching: Create a mask for dij < 1.0
            auto mask = dij_sq < logic_threshold;

            // Calculate the "else" path: 10.0 / (dist^3)
            // Using rsqrt for speed: inv_dist = 1/sqrt(dij_sq)
            b_type inv_dist = xs::rsqrt(dij_sq);
            b_type inv_dist3 = inv_dist * inv_dist * inv_dist;
            b_type else_val = logic_default * inv_dist3;

            // Select: if mask is true, use 10.0, else use else_val
            // Note: we also handle the i == j case implicitly here 
            // as dij_sq will be 0, which is < 1.0, so it gets 10.0. 
            // But since diff is 0, adding 0 * 10 has no effect.
            b_type final_dij = xs::select(mask, logic_default, else_val);

            raccx_i += diffx * final_dij * mass_j;
            raccy_i += diffy * final_dij * mass_j;
            raccz_i += diffz * final_dij * mass_j;
        }

        raccx_i.store_unaligned(&accelerationsx[i]);
        raccy_i.store_unaligned(&accelerationsy[i]);
        raccz_i.store_unaligned(&accelerationsz[i]);
    }

    // 3. Update Velocities and Positions (Integration)
    // This part is also vectorized for extra speed
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