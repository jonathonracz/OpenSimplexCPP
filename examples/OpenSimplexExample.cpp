/* This is free and unencumbered software released into the public domain.
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

#include <vector>
#include <fstream>

#include "OpenSimplex/OpenSimplex.h"

#pragma pack(push, 1)
struct TGAHeader {
    char idLength = 0;
    char colorMapType = 0;
    char dataTypeCode = 2;
    short colorMapOrigin = 0;
    short colorMapLength = 0;
    char colorMapDepth = 0;
    short xOrigin = 0;
    short yOrigin = 0;
    short width;
    short height;
    char bitsPerPixel = 32;
    char imageDescriptor = 8;
};
#pragma pack(pop)

static_assert(sizeof(TGAHeader) == 18, "The TGA file header MUST be 18 bytes - make sure your compiler is packing the struct.");

static void write_tga_image(std::string filename, uint32_t* pixels, int w, int h)
{
    std::vector<char> imageFile(sizeof(TGAHeader) + (w * h * sizeof(uint32_t)));

    TGAHeader header;
    header.width = w;
    header.height = h;

    memcpy(imageFile.data(), &header, sizeof(header));
    memcpy(&imageFile[sizeof(TGAHeader)], pixels, header.width * header.height * sizeof(uint32_t));

    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (file.is_open())
        file.write(imageFile.data(), imageFile.size());
}

int main(int argc, char* argv[])
{
    const int width = 512;
    const int height = 512;
    const int featureSize = 24;

    uint32_t image2d[width * height];
    uint32_t image3d[width * height];
    uint32_t image4d[width * height];

    OpenSimplex::Context context;
    OpenSimplex::Seed::computeContextForSeed(context, 77374);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float value;
#if defined(SINGLE_OCTAVE)
            value = openSimplex.noise4((float) x / featureSize, (float) y / featureSize, 0.0, 0.0);
#else
            /* Use three octaves: frequency N, N/2 and N/4 with relative amplitudes 4:2:1. */
            float v0 = OpenSimplex::Noise::noise4(context, (float) x / featureSize / 4,
                        (float) y / featureSize / 4, 0.0, 0.0);
            float v1 = OpenSimplex::Noise::noise4(context, (float) x / featureSize / 2,
                        (float) y / featureSize / 2, 0.0, 0.0);
            float v2 = OpenSimplex::Noise::noise4(context, (float) x / featureSize / 1,
                        (float) y / featureSize / 1, 0.0, 0.0);
            value = v0 * 4 / 7.0 + v1 * 2 / 7.0 + v2 * 1 / 7.0;
#endif
            uint32_t rgb = 0x010101 * (uint32_t) ((value + 1) * 127.5);
            image2d[(y * width) + x] = (0x0ff << 24) | (rgb);

            value = OpenSimplex::Noise::noise2(context, (float) x / featureSize, (float) y / featureSize);
            rgb = 0x010101 * (uint32_t) ((value + 1) * 127.5);
            image3d[(y * width) + x] = (0x0ff << 24) | (rgb);

            value = OpenSimplex::Noise::noise3(context, (float) x / featureSize, (float) y / featureSize, 0.0);
            rgb = 0x010101 * (uint32_t) ((value + 1) * 127.5);
            image4d[(y * width) + x] = (0x0ff << 24) | (rgb);
        }
    }

    write_tga_image("test2d.tga", image2d, width, height);
    write_tga_image("test3d.tga", image3d, width, height);
    write_tga_image("test4d.tga", image4d, width, height);

    return 0;
}
