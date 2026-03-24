//ESTE CODIGO REDUCE LAS INTERACCIONES DE N^2 A N^2/2, APROVECHANDO LA SIMETRIA DE LAS FUERZAS ENTRE PARES DE PARTICULAS. Pero baja un poco el rendimiento, probablemente por la complejidad añadida y la falta de paralelismo en la fase B. Se mantiene como referencia de optimización futura.

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

    // 1. Setup Thread-Local Buffers
    int max_threads = omp_get_max_threads();
    
    // Allocate private memory for each thread. 
    // Using std::vector handles cleanup automatically.
    std::vector<std::vector<float>> local_accx(max_threads, std::vector<float>(n_particles, 0.0f));
    std::vector<std::vector<float>> local_accy(max_threads, std::vector<float>(n_particles, 0.0f));
    std::vector<std::vector<float>> local_accz(max_threads, std::vector<float>(n_particles, 0.0f));

    // 2. Compute Accelerations (Reduced to N^2 / 2 interactions)
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // Use schedule(dynamic) because workload per i-iteration decreases as i grows
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < n_particles; i += b_type::size)
        {
            const b_type rposx_i = b_type::load_unaligned(&particles.x[i]);
            const b_type rposy_i = b_type::load_unaligned(&particles.y[i]);
            const b_type rposz_i = b_type::load_unaligned(&particles.z[i]);
            
            // We need the mass of the 'i' batch to compute the force on 'j' later
            const b_type mass_i  = b_type::load_unaligned(&initstate.masses[i]);

            b_type raccx_i(0.0f);
            b_type raccy_i(0.0f);
            b_type raccz_i(0.0f);

            // Phase A: Intra-block interactions (interactions WITHIN this SIMD batch)
            // To keep SIMD clean, we compute these normally without symmetry.
            int j_simd_end = std::min(i + (int)b_type::size, n_particles);
            for (int j = i; j < j_simd_end; ++j)
            {
                const b_type posx_j(particles.x[j]);
                const b_type posy_j(particles.y[j]);
                const b_type posz_j(particles.z[j]);
                const b_type mass_j(initstate.masses[j]);

                const b_type diffx = posx_j - rposx_i;
                const b_type diffy = posy_j - rposy_i;
                const b_type diffz = posz_j - rposz_i;

                b_type dij_sq = (diffx * diffx) + (diffy * diffy) + (diffz * diffz);
                auto mask = dij_sq < logic_threshold;
                
                b_type inv_dist = xs::rsqrt(dij_sq);
                b_type inv_dist3 = inv_dist * inv_dist * inv_dist;
                b_type else_val = logic_default * inv_dist3;

                b_type final_dij = xs::select(mask, logic_default, else_val);

                // Only update i here. j will get its update when it is processed as an i.
                raccx_i += diffx * final_dij * mass_j;
                raccy_i += diffy * final_dij * mass_j;
                raccz_i += diffz * final_dij * mass_j;
            }

            // Phase B: Inter-block interactions (apply Newton's Third Law)
            // Start j AFTER the current SIMD block
            for (int j = j_simd_end; j < n_particles; ++j)
            {
                const b_type posx_j(particles.x[j]);
                const b_type posy_j(particles.y[j]);
                const b_type posz_j(particles.z[j]);
                const b_type mass_j(initstate.masses[j]);

                const b_type diffx = posx_j - rposx_i;
                const b_type diffy = posy_j - rposy_i;
                const b_type diffz = posz_j - rposz_i;

                b_type dij_sq = (diffx * diffx) + (diffy * diffy) + (diffz * diffz);
                auto mask = dij_sq < logic_threshold;

                b_type inv_dist = xs::rsqrt(dij_sq);
                b_type inv_dist3 = inv_dist * inv_dist * inv_dist;
                b_type else_val = logic_default * inv_dist3;

                b_type final_dij = xs::select(mask, logic_default, else_val);

                // 1. Update i's acceleration (Vectorized)
                raccx_i += diffx * final_dij * mass_j;
                raccy_i += diffy * final_dij * mass_j;
                raccz_i += diffz * final_dij * mass_j;

                // 2. Update j's acceleration (Symmetry / Third Law)
                // Acceleration on j is opposite direction (-diff). 
                // We use reduce_add to sum the batch of i forces horizontally down to a scalar.
                local_accx[tid][j] -= xs::hadd(diffx * final_dij * mass_i);
                local_accy[tid][j] -= xs::hadd(diffy * final_dij * mass_i);
                local_accz[tid][j] -= xs::hadd(diffz * final_dij * mass_i);
            }

            // Store the computed SIMD accelerations for i into this thread's local buffer
            float temp_ax[b_type::size], temp_ay[b_type::size], temp_az[b_type::size];
            raccx_i.store_unaligned(temp_ax);
            raccy_i.store_unaligned(temp_ay);
            raccz_i.store_unaligned(temp_az);
            
            for (int k = 0; k < b_type::size && (i + k) < n_particles; ++k) {
                local_accx[tid][i + k] += temp_ax[k];
                local_accy[tid][i + k] += temp_ay[k];
                local_accz[tid][i + k] += temp_az[k];
            }
        }
    }

    // 3. Final Reduction Step
    // Sum all the thread-local buffers directly into the global arrays
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_particles; ++i)
    {
        float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
        for (int t = 0; t < max_threads; ++t)
        {
            sum_x += local_accx[t][i];
            sum_y += local_accy[t][i];
            sum_z += local_accz[t][i];
        }
        
        // This overwrites the old accelerations, replacing the need for std::fill!
        accelerationsx[i] = sum_x;
        accelerationsy[i] = sum_y;
        accelerationsz[i] = sum_z;
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