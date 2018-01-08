#pragma once

#include "Environment.h"
#include "Context.h"

namespace OpenSimplex
{
    class Noise
    {
    public:
        static float noise2(OPENSIMPLEX_GPU_CONSTANT const Context& context, float x, float y);
        static float noise3(OPENSIMPLEX_GPU_CONSTANT const Context& context, float x, float y, float z);
        static float noise4(OPENSIMPLEX_GPU_CONSTANT const Context& context, float x, float y, float z, float w);

        static OPENSIMPLEX_GPU_CONSTANT const int8_t gradients2D[16];
        static OPENSIMPLEX_GPU_CONSTANT const int8_t gradients3D[72];
        static OPENSIMPLEX_GPU_CONSTANT const int8_t gradients4D[256];

    private:
        static float floor(float x);
        static float extrapolate2(OPENSIMPLEX_GPU_CONSTANT const Context& context, int xsb, int ysb, float dx, float dy);
        static float extrapolate3(OPENSIMPLEX_GPU_CONSTANT const Context& context, int xsb, int ysb, int zsb, float dx, float dy, float dz);
        static float extrapolate4(OPENSIMPLEX_GPU_CONSTANT const Context& context, int xsb, int ysb, int zsb, int wsb, float dx, float dy, float dz, float dw);
    };
}
