# OpenSimplexCPP

This is a header-only, dependency-free, GPU compute ready implementation of OpenSimplex Noise derived from Stephen M. Cameron's C port of Kurt Spencer's Java implementation.

The goal is to produce an OpenSimplex implementation that does not perform any dynamic allocation and has no dependencies on standard headers outside fixed-width types. This not only makes it portable across operating systems but enables it to run in most GPU compute environments (CUDA, Metal, OpenCL C++).

As always, issues and pull requests are welcome.

## Usage
In your CPU code, compute a seeded context:

```c++
#include <OpenSimplex/OpenSimplex.h>

...

OpenSimplex::Context ctx;
OpenSimplex::Seed::computeContextForSeed(ctx, 42);
```

Add the `OpenSimplex::Context` object to your uniform buffer (take note that it's 512 bytes).

In your shader, use the `OpenSimplex::Context` to compute per-fragment noise (note this example is in Metal - other languages will be very similar):

```c++
using namespace metal;

#include "OpenSimplex/OpenSimplex.h"

kernel void kernelMain(texture2d<half, access::write> output [[texture0]],
                    constant Uniforms& uniforms [[buffer0]],
                    simd::uint2 gid [[thread_position_in_grid]])
{
    float value = OpenSimplex::Noise::noise2(uniforms.context, gid.x, gid.y);

    output.write(half4(value, value, value, 1.0f), gid);
}
```

![Screenshot](/examples/screenshot.png?raw=true)
