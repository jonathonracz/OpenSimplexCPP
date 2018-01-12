/*
 * OpenSimplex (Simplectic) Noise in portable GPGPU-compatible C++.
 * Derived from Stephen M. Cameron's C port of Kurt Spencer's Java
 * implementation by Jonathon Racz.
 *
 * This is free and unencumbered software released into the public domain.
 *
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 *
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * For more information, please refer to <http://unlicense.org>
 */

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
    inline void computeContextForSeed(Context& context, int64_t seed);
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
