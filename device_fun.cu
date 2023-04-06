#include <iostream>
#include <array>
#include "vect_fun.cu"

#define distr_range(random_fun, min, max) (random_fun * (max - (min))) + min

const unsigned int BLOCK_SIZE = 512;

__global__ void block_agregation_kernel(
    unsigned int *global_mem,
    unsigned int *result)
{
    for (unsigned int s = gridDim.x / 2; s > 0; s >>= 1)
    {
        if (blockIdx.x < s)
        {
            global_mem[blockIdx.x] += global_mem[blockIdx.x + s];
        }
        __syncthreads();
    }
    cudaMemcpyAsync(result,
                    global_mem,
                    sizeof(unsigned int),
                    cudaMemcpyDeviceToDevice);
}

__global__ void volume_tetrahedron_on_device(
    const float3 &A,
    const float3 &B,
    const float3 &C,
    const float3 &D,
    unsigned int N,
    unsigned int *accumulator,
    unsigned int *global_mem)
{
    int tid = threadIdx.x;
    extern __shared__ unsigned int block_count[];
    curandState state;
    curand_init(clock64(), tid, 0, &state);
    float3 P{static_cast<float>(distr_range(curand_uniform(&state), -1.5, 0.3)),
             static_cast<float>(distr_range(curand_uniform(&state), -0.2, 0.4)),
             static_cast<float>(distr_range(curand_uniform(&state), -0.7, 0.5))};

    if (inside_tetrahedron(A, B, C, D, P))
    {
        atomicAdd(&global_mem[blockIdx.x], 1U);
    }
    __syncthreads();

    if (tid == 0 && blockIdx.x == 0)
    {
        cudaStream_t stream;
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        dim3 childGrid{gridDim.x};
        dim3 childBlock{1};
        block_agregation_kernel<<<childGrid, childBlock, 0, stream>>>(global_mem, accumulator);
    }
    __syncthreads();
}

unsigned long long device_estimate(
    const std::array<float3, 4> &vertices,
    const unsigned long long &N)
{
    int count = 0;
    int block_size = BLOCK_SIZE;
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    unsigned int *accumulator;
    cudaMalloc(&accumulator, sizeof(unsigned int));
    cudaMemset(accumulator, 0, sizeof(unsigned int));

    unsigned int *global_mem;
    cudaMallocManaged(&global_mem, sizeof(unsigned int) * grid_size);
    cudaMemset(global_mem, 0, sizeof(unsigned int));

    float3 *dev_vertices;
    cudaMallocManaged(&dev_vertices, sizeof(float3) * 4);
    cudaMemcpy(dev_vertices, vertices.data(), sizeof(float3) * 4, cudaMemcpyHostToDevice);

    volume_tetrahedron_on_device<<<grid_size, block_size, sizeof(unsigned int)>>>(
        dev_vertices[0],
        dev_vertices[1],
        dev_vertices[2],
        dev_vertices[3],
        N,
        accumulator,
        global_mem);

    cudaMemcpy(&count, accumulator, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    return count;
}