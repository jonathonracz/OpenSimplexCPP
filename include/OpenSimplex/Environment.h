#pragma once

#if defined(__AIR64__) || defined(OPENCL_COMPILER) || defined(__NVCC__)
    #define OPENSIMPLEX_IS_GPU 1
#else
    #define OPENSIMPLEX_IS_GPU 0
#endif

#if !OPENSIMPLEX_IS_GPU
    #include <cstdint> /* For fixed width types. */
#endif

#if defined(__AIR64__)
    #define OPENSIMPLEX_GPU_CONSTANT constant
#elif defined(OPENCL_COMPILER)
    #define OPENSIMPLEX_GPU_CONSTANT __constant
#elif defined(__NVCC__)
    #define OPENSIMPLEX_GPU_CONSTANT __constant__
#else
    #define OPENSIMPLEX_GPU_CONSTANT
#endif
