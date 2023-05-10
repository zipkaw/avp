#pragma once
#include "funs.cu"

unsigned int collect_accumulator(unsigned int *accum, const int min_side)
{
    unsigned int max_R = accum[0];
    for (int i = 1; i < min_side; i++)
    {
        if (max_R < accum[i])
        {
            max_R = i;
        }
    }
    return max_R;
}

__global__ void hought_transform(unsigned char *src, size_t width, size_t height, size_t channels, size_t pitch, unsigned int *accumulator)
{
    const unsigned x = (threadIdx.x + blockDim.x * blockIdx.x);
    const unsigned y = threadIdx.y + blockDim.y * blockIdx.y;

    unsigned int x0 = height / 2;
    unsigned int y0 = width / 2;
    unsigned int R = 0;

    if (!pixel_check(src, (x * width + y) * channels))
    {
        return;
    }
    R = static_cast<int>(ceil(sqrt(static_cast<float>(((x - x0) * (x - x0)) + ((y - y0) * (y - y0))))));
    atomicAdd(accumulator + R, 1U);
}

__global__ void fish_eye(uint8_t *image, uint8_t *output, const int width,
                         const int height, int channels, const int radius,
                         const float coefficient, const int out_pitch, const int im_pitch, bool *mask)
{
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;
    const unsigned int tid = threadIdx.x * blockDim.y + threadIdx.y;
    extern __shared__ RGBA_t pixel[];
    if (y > width && x > height)
    {
        return;
    }
    pixel[tid] = RGBA_t(image, x, y, im_pitch, height, channels);
    const float e = 0.001f;
    const float x0 = normalize_value(static_cast<float>(x), static_cast<float>(height));
    const float y0 = normalize_value(static_cast<float>(y), static_cast<float>(width));
    const float r = sqrt(x0 * x0 + y0 * y0);
    const float teta = atan2(y0, x0);
    const float scale = min(1.0f / abs(cos(teta) + e), 1.0f / abs(sin(teta) + e));
    const float _r = min(scale, 1) * pow(r, coefficient);

    auto _x = static_cast<float>(height) / 2.0f * _r * cos(teta) +
              (static_cast<float>(height) + 1.0f) / 2.0f;
    auto _y = static_cast<float>(width) / 2.0f * _r * sin(teta) +
              (static_cast<float>(width) + 1.0f) / 2.0f;

    if (static_cast<unsigned int>(_x) < height && _x >= 0 && static_cast<unsigned int>(_y) < width && _y >= 0)
    {
        // const RGBA output_pixel = interpolation(image, _x, _y, x, y, width, height, channels, im_pitch);
        const RGBA output_pixel = shared_interpolation(pixel, _x, _y, threadIdx.x, threadIdx.y, width, height);
        output[(static_cast<int>(round(_x)) * width + static_cast<int>(round(_y))) * channels] = output_pixel.r;
        output[(static_cast<int>(round(_x)) * width + static_cast<int>(round(_y))) * channels + 1] = output_pixel.g;
        output[(static_cast<int>(round(_x)) * width + static_cast<int>(round(_y))) * channels + 2] = output_pixel.b;
        output[(static_cast<int>(round(_x)) * width + static_cast<int>(round(_y))) * channels + 3] = output_pixel.a;
        mask[(static_cast<int>(round(_x))) * width + (static_cast<int>(round(_y)))] = true;
    }
}