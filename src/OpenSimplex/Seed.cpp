#include "OpenSimplex/Seed.h"
#include "OpenSimplex/Noise.h"

namespace OpenSimplex
{
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
            context.permGradIndex3D[i] = (short)((context.perm[i] % ((sizeof((Noise::gradients3D)) / sizeof((Noise::gradients3D)[0])) / 3)) * 3);
            source[r] = source[i];
        }
    }
}
