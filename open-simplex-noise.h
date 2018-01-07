#pragma once

/*
 * OpenSimplex (Simplectic) Noise in C.
 * Ported to C from Kurt Spencer's java implementation by Stephen M. Cameron
 *
 * v1.1 (October 6, 2014) 
 * - Ported to C
 * 
 * v1.1 (October 5, 2014)
 * - Added 2D and 4D implementations.
 * - Proper gradient sets for all dimensions, from a
 *   dimensionally-generalizable scheme with an actual
 *   rhyme and reason behind it.
 * - Removed default permutation array in favor of
 *   default seed.
 * - Changed seed-based constructor to be independent
 *   of any particular randomization library, so results
 *   will be the same when ported to other languages.
 */

#if !defined(__AIR64__) && !defined(OPENCL_COMPILER) && !defined(__NVCC__)
#include <cstdlib> // Needed for fixed width types.
#endif

struct osn_context {
	int16_t perm[256];
	int16_t permGradIndex3D[256];
};

int open_simplex_noise(int64_t seed, osn_context *ctx);
float open_simplex_noise2(osn_context *ctx, float x, float y);
float open_simplex_noise3(osn_context *ctx, float x, float y, float z);
float open_simplex_noise4(osn_context *ctx, float x, float y, float z, float w);

//class 
