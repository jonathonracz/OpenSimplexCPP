# OpenSimplexCPP

This is a single-header library containing a portable, dependency-free implementation of OpenSimplex Noise. It is derived from Stephen M. Cameron's C port of Kurt Spencer's Java implementation.

The goal is to produce an OpenSimplex implementation that does not perform any dynamic allocation and has no dependencies on standard headers. This not only makes it portable across operating systems but enables it to run in most GPU compute environments (CUDA, Metal, OpenCL C++).

Usage is extremely simple:

```c++
#include "OpenSimplex/OpenSimplex.h"

int main(int argc, char** argv)
{
    OpenSimplex osimp(42); // Arbitrary seed
    float value = osimp.noise2(7, 12); // Arbitrary x/y coordinate
}
```

As always, issues and pull requests are welcome.
