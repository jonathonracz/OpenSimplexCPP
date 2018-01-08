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
        /*
         * Initializes using a permutation array generated from a 64-bit seed.
         * Generates a proper permutation (i.e. doesn't merely perform N
         * successive pair swaps on a base array). Uses a simple 64-bit LCG.
         */
        void computeContextForSeed(Context& context, int64_t seed);
    }
}
