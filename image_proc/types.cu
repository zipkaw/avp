#pragma once
#include <stdint.h>
#include <math.h>
#include <string.h>

#define MARKER 0x00FF00 // green color

constexpr int PIXELS_PER_THREAD = 4;
constexpr dim3 BLOCK_DIM{16, 16};

#define min(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

typedef struct RGBA_t
{
    uint8_t r, g, b, a;

    __device__
    RGBA_t(uint8_t *image, const int x, const int y, const int width, const int height, const int channels)
    {
        r = image[x * width + y * channels];
        g = image[x * width + y * channels + 1];
        b = image[x * width + y * channels + 2];
        a = image[x * width + y * channels + 3];
    }
    __device__
    RGBA_t(uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        this->r = r;
        this->g = g;
        this->b = b;
        this->a = a;
    }

    uint8_t dot_product(RGBA_t a, RGBA_t b)
    {
        return a.r * b.r + a.g * b.g + a.b * b.b + a.a * b.a;
    }

} RGBA;

typedef struct pixel_t
{
    int2 T;
    RGBA C;
} pixel;

__device__
    int2
    operator-(int2 a, int2 b)
{
    return {a.x - b.x, a.y - b.y};
}
__device__
    float2
    operator-(float2 a, float2 b)
{
    return {a.x - b.x, a.y - b.y};
}
__device__ float operator/(float2 a, float2 b)
{
    return a.x / b.x +
           a.y / b.y;
}
__device__
    RGBA
    operator*(RGBA a, float b)
{
    a.r = static_cast<uint8_t>(static_cast<float>(a.r) * b);
    a.g = static_cast<uint8_t>(static_cast<int>(a.g) * b);
    a.b = static_cast<uint8_t>(static_cast<int>(a.b) * b);
    a.a = static_cast<uint8_t>(static_cast<int>(a.a) * b);
    return a;
}

__device__
    RGBA
    operator+(RGBA a, RGBA b)
{
    a.r = a.r + b.r;
    a.g = a.g + b.g;
    a.b = a.b + b.b;
    a.a = a.a + b.a;
    return a;
}