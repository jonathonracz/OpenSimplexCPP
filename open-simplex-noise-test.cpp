#include <vector>
#include <fstream>

#include "open-simplex-noise.h"

#define WIDTH 512
#define HEIGHT 512
#define FEATURE_SIZE 24

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
    int x, y;
    float value;
    float v0, v1, v2; /* values from different octaves. */
    uint32_t rgb;
    uint32_t image2d[WIDTH * HEIGHT];
    uint32_t image3d[WIDTH * HEIGHT];
    uint32_t image4d[WIDTH * HEIGHT];

    OpenSimplex openSimplex(77374);

    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH; x++) {
#if defined(SINGLE_OCTAVE)
            value = openSimplex.noise4((float) x / FEATURE_SIZE, (float) y / FEATURE_SIZE, 0.0, 0.0);
#else
            /* Use three octaves: frequency N, N/2 and N/4 with relative amplitudes 4:2:1. */
            v0 = openSimplex.noise4((float) x / FEATURE_SIZE / 4,
                        (float) y / FEATURE_SIZE / 4, 0.0, 0.0);
            v1 = openSimplex.noise4((float) x / FEATURE_SIZE / 2,
                        (float) y / FEATURE_SIZE / 2, 0.0, 0.0);
            v2 = openSimplex.noise4((float) x / FEATURE_SIZE / 1,
                        (float) y / FEATURE_SIZE / 1, 0.0, 0.0);
            value = v0 * 4 / 7.0 + v1 * 2 / 7.0 + v2 * 1 / 7.0;
#endif
            rgb = 0x010101 * (uint32_t) ((value + 1) * 127.5);
            image2d[(y * WIDTH) + x] = (0x0ff << 24) | (rgb);

            value = openSimplex.noise2((float) x / FEATURE_SIZE, (float) y / FEATURE_SIZE);
            rgb = 0x010101 * (uint32_t) ((value + 1) * 127.5);
            image3d[(y * WIDTH) + x] = (0x0ff << 24) | (rgb);

            value = openSimplex.noise3((float) x / FEATURE_SIZE, (float) y / FEATURE_SIZE, 0.0);
            rgb = 0x010101 * (uint32_t) ((value + 1) * 127.5);
            image4d[(y * WIDTH) + x] = (0x0ff << 24) | (rgb);
        }
    }
    write_tga_image("test2d.tga", image2d, WIDTH, HEIGHT);
    write_tga_image("test3d.tga", image3d, WIDTH, HEIGHT);
    write_tga_image("test4d.tga", image4d, WIDTH, HEIGHT);
    return 0;
}
