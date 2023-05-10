#pragma once
#include "funs.cu"

__device__ void init_above(RGBA_t *shared, const uint2 shared_dim,
                           uint8_t *image, const int x, const int y,
                           const int width, const int height, const int pitch, const int channels)
{
    for (int i = 0; i < shared_dim.y - 1; i++)
    {
        if ((x - 1) > 0 && ((y - 1 + i) > 0 && (y - 1 + i) < width))
        {
            shared[i] = RGBA_t(image, x - 1, y - 1 + i, pitch, height, channels);
        }
        else
        {
            shared[i] = RGBA{image, x, y, pitch, height, channels};
            // shared[i] = RGBA{255, 255, 255, 255};
        }
    }
}

__device__ void init_right(RGBA_t *shared, const uint2 shared_dim,
                           uint8_t *image, const int x, const int y,
                           const int width, const int height, const int pitch, const int channels)
{
    for (int i = 0; i < shared_dim.x - 1; i++)
    {
        if (((x - 1 + i) > 0 && (x - 1 + i) < height) && (y + 1) < width)
        {
            shared[i * shared_dim.y + shared_dim.y - 1] = RGBA_t(image, x - 1 + i, y + 1, pitch, height, channels);
        }
        else
        {
            shared[i * shared_dim.y + shared_dim.y - 1] = RGBA{image, x, y, pitch, height, channels};
        }
    }
}

__device__ void init_bottom(RGBA_t *shared, const uint2 shared_dim,
                            uint8_t *image, const int x, const int y,
                            const int width, const int height, const int pitch, const int channels)
{
    for (int i = shared_dim.y - 1; i > 1; i--)
    {
        if ((x + 1) < height && (y + 1 + i) < width)
        {
            shared[(shared_dim.y - 1) * shared_dim.y + i] = RGBA_t(image, x + 1, y + 1 + i, pitch, height, channels);
        }
        else
        {
            shared[(shared_dim.y - 1) * shared_dim.y + i] = RGBA{image, x, y, pitch, height, channels};
        }
    }
}

__device__ void init_left(RGBA_t *shared, const uint2 shared_dim,
                          uint8_t *image, const int x, const int y,
                          const int width, const int height, const int pitch, const int channels)
{
    for (int i = shared_dim.x - 1; i > 1; i--)
    {
        if ((x + 1 + i) < height && ((y - 1) > 0))
        {
            shared[i * shared_dim.y] = RGBA_t(image, x + 1 + i, y - 1, pitch, height, channels);
        }
        else
        {
            shared[i * shared_dim.y] = RGBA{image, x, y, pitch, height, channels};
        }
    }
}

__device__ void init_shared(RGBA_t *shared, bool *mask, uint8_t *image,
                            const unsigned int tidX, const unsigned int tidY,
                            const int x, const int y,
                            const int width, const int height, const int pitch, const int channels)
{
    const uint2 pos00{1, 1};
    const uint2 pos10{BLOCK_DIM.x, 1};
    const uint2 pos01{1, BLOCK_DIM.y};
    const uint2 pos11{BLOCK_DIM.x, BLOCK_DIM.y};

    const uint2 dim_shared{BLOCK_DIM.x + 1, BLOCK_DIM.y + 1};
    const uint2 pos_in_shared{tidX + 1, tidY + 1};

    shared[pos_in_shared.x * dim_shared.y + pos_in_shared.y] = RGBA_t(image, x, y, pitch, height, channels);
    if (tidX == pos00.x && tidY == pos00.y)
    {
        init_above(shared, dim_shared, image, x, y, width, height, pitch, channels);
    }
    else if (tidX == pos10.x && tidY == pos10.y)
    {
        init_right(shared, dim_shared, image, x, y, width, height, pitch, channels);
    }
    else if (tidX == pos01.x && tidY == pos01.y)
    {
        init_left(shared, dim_shared, image, x, y, width, height, pitch, channels);
    }
    else if (tidX == pos11.x && tidY == pos11.y)
    {
        init_bottom(shared, dim_shared, image, x, y, width, height, pitch, channels);
    }
}

__global__ void recover(uint8_t *image, const int width,
                        const int height, int channels,
                        const int im_pitch, bool *mask)
{
    const unsigned int x = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned int y = threadIdx.y + blockDim.y * blockIdx.y;

    const unsigned int tid = (threadIdx.x + 1) * (blockDim.y + 1) + (threadIdx.y + 1);
    extern __shared__ RGBA_t valid_pixels[];

    if (y > width && x > height)
    {
        return;
    }

    valid_pixels[tid] = RGBA_t(image, x, y, im_pitch, height, channels);
    if (mask[x * width + y] == false)
    {
        const RGBA output_pixel = shared_interpolation(image, mask, x, y, width, height, im_pitch, channels);
        image[(x * width + y) * channels] =     output_pixel.r;
        image[(x * width + y) * channels + 1] = output_pixel.g;
        image[(x * width + y) * channels + 2] = output_pixel.b;
        image[(x * width + y) * channels + 3] = output_pixel.a;

    }
}
