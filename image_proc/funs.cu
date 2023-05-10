#pragma once
#include "types.cu"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ bool pixel_check(uint8_t *image, size_t posx)
{
    if ((image[posx] << 16) +
            (image[posx + 1] << 8) +
            image[posx + 2] ==
        MARKER)
    {
        return true;
    }
    return false;
}

__host__ __device__ float normalize_value(const float value, const float max)
{
    return (value - ((max + 1.0f) / 2.0f)) * 2.0f / (max);
}

float calc_coefficient(const float width, const float height, const float radius)
{
    const float normalized_source_circle_pointX = normalize_value(height / 2.0f, height);
    const float normalized_source_circle_pointY = normalize_value(width / 2.0f + radius, width);
    const float normalized_source_circle_radius = sqrt(normalized_source_circle_pointX * normalized_source_circle_pointX +
                                                       normalized_source_circle_pointY * normalized_source_circle_pointY);

    const float normalized_target_circle_pointY = normalize_value(width / 2.0f, width);
    const float normalized_target_circle_pointX = normalize_value(height / 2.0f + min(width, height) * 0.1f, height);
    const float normalized_target_circle_radius = sqrt(normalized_target_circle_pointX * normalized_target_circle_pointX +
                                                       normalized_target_circle_pointY * normalized_target_circle_pointY);

    return log(normalized_target_circle_radius) / log(normalized_source_circle_radius);
}

__device__
    RGBA // bilenear interpolation for float point
    interpolation(uint8_t *input, const float _x, const float _y, const int x, const int y, const int width, const int height, const int channels, const int pitch)
{
    const int x0 = x;
    const int y0 = y;
    const int x1 = x + 1 < height ? x + 1 : x;
    const int y1 = y + 1 < width ? y + 1 : y;

    const float local_x = _x - trunc(_x);
    const float local_y = _y - trunc(_y);

    RGBA c00(input, x0, y1, pitch, height, channels);
    RGBA c10(input, x1, y0, pitch, height, channels);
    RGBA c01(input, x0, y1, pitch, height, channels);
    RGBA c11(input, x1, y1, pitch, height, channels);

    RGBA c{c00 * ((1 - local_x) * (1 - local_y)) +
           c10 * (local_x * (1 - local_y)) +
           c01 * (local_y * (1 - local_x)) +
           c11 * (local_x * local_y)};

    c.a = 255;
    return c;
}

__device__
    RGBA // bilenear interpolation for float point using shared memory
    shared_interpolation(RGBA_t *pixel,
                         const float _x, const float _y,
                         const int x, const int y,
                         const int width, const int height)
{
    const int x0 = x;
    const int y0 = y;
    const int x1 = x + 1 < BLOCK_DIM.x ? x + 1 : x;
    const int y1 = y + 1 < BLOCK_DIM.y ? y + 1 : y;

    const float local_x = _x - trunc(_x);
    const float local_y = _y - trunc(_y);

    RGBA c00 = pixel[x0 * BLOCK_DIM.y + y0];
    RGBA c10 = pixel[x1 * BLOCK_DIM.y + y0];
    RGBA c01 = pixel[x0 * BLOCK_DIM.y + y1];
    RGBA c11 = pixel[x1 * BLOCK_DIM.y + y1];

    RGBA c{c00 * ((1 - local_x) * (1 - local_y)) +
           c10 * (local_x * (1 - local_y)) +
           c01 * (local_y * (1 - local_x)) +
           c11 * (local_x * local_y)};

    c.a = 255;
    return c;
}

__device__ float agregate_coeff(uint4 mask)
{
    if ((mask.x + mask.y + mask.z + mask.w) == 0)
    {
        return 0;
    }
    else
    {
        return 1.0f / static_cast<float>(mask.x + mask.y + mask.z + mask.w);
    }
};

__device__
    RGBA // bilinear interpolation for integer point
    shared_interpolation(uint8_t *image, bool *image_mask,
                         const int x, const int y,
                         const int width, const int height, const int pitch, const int channels)
{
    const int x0 = x - 1 > 0 ? x - 1 : x;
    const int y0 = y - 1 > 0 ? y - 1 : y;
    const int x1 = x + 1 < height ? x + 1 : x;
    const int y1 = y + 1 < width ? y + 1 : y;

    uint4 mask{1, 1, 1, 1};
    RGBA cx{image[(x * width + y) * channels],
            image[(x * width + y) * channels + 1],
            image[(x * width + y) * channels + 2],
            image[(x * width + y) * channels + 3]};

    RGBA c00{image[(x0 * width + y0) * channels],
             image[(x0 * width + y0) * channels + 1],
             image[(x0 * width + y0) * channels + 2],
             image[(x0 * width + y0) * channels + 3]};

    RGBA c10{image[(x1 * width + y0) * channels],
             image[(x1 * width + y0) * channels + 1],
             image[(x1 * width + y0) * channels + 2],
             image[(x1 * width + y0) * channels + 3]};

    RGBA c01{image[(x0 * width + y1) * channels],
             image[(x0 * width + y1) * channels + 1],
             image[(x0 * width + y1) * channels + 2],
             image[(x0 * width + y1) * channels + 3]};

    RGBA c11{image[(x1 * width + y1) * channels],
             image[(x1 * width + y1) * channels + 1],
             image[(x1 * width + y1) * channels + 2],
             image[(x1 * width + y1) * channels + 3]};

    if ((c00.a + c00.b + c00.g + c00.r) == 0)
    {
        mask.x = 0;
    }
    if ((c10.a + c10.b + c10.g + c10.r) == 0)
    {
        mask.y = 0;
    }
    if ((c01.a + c01.b + c01.g + c01.r) == 0)
    {
        mask.z = 0;
    }
    if ((c11.a + c11.b + c11.g + c11.r) == 0)
    {
        mask.w = 0;
    }

    const float coefficient = agregate_coeff(mask);

    RGBA c{c00 * mask.x * coefficient +
           c10 * mask.y * coefficient +
           c01 * mask.z * coefficient +
           c11 * mask.w * coefficient};
    c.a = 255;
    return c;
}