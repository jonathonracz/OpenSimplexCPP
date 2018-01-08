#pragma once

#include "Environment.h"

#if OPENSIMPLEX_IS_GPU
    #error "Seed shouldn't be running on the GPU - so don't try including it!"
#endif

#include "Context.h"

namespace OpenSimplex
{

namespace Seed
{
    void computeContextForSeed(Context& context, int64_t seed);
}

/*
 * Initializes using a permutation array generated from a 64-bit seed.
 * Generates a proper permutation (i.e. doesn't merely perform N
 * successive pair swaps on a base array). Uses a simple 64-bit LCG.
 */
void Seed::computeContextForSeed(OpenSimplex::Context& context, int64_t seed)
{
    int16_t source[256];

    for (int i = 0; i < 256; i++)
        source[i] = (int16_t) i;

    seed = seed * 6364136223846793005LL + 1442695040888963407LL;
    seed = seed * 6364136223846793005LL + 1442695040888963407LL;
    seed = seed * 6364136223846793005LL + 1442695040888963407LL;

    for (int i = 255; i >= 0; i--) {
        seed = seed * 6364136223846793005LL + 1442695040888963407LL;
        int r = (int)((seed + 31) % (i + 1));
        if (r < 0)
            r += (i + 1);
        context.perm[i] = source[r];
        // Note that "72" is the number of entries in the 3D gradient array.
        context.permGradIndex3D[i] = (short)((context.perm[i] % (72 / 3)) * 3);
        source[r] = source[i];
    }
}

}
