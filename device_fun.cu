#include <iostream>
#include <array>
#include "vect_fun.cu"

#define distr_range(random_fun, min, max) (random_fun * (max - (min))) + min

constexpr unsigned int BLOCK_SIZE = 512;
constexpr unsigned int ITER_PER_BLOCK = 100;

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x;
    int count_inside = 0;
    extern __shared__ unsigned int s_count[];

    curandState state;
    curand_init(clock64(), tid, 0, &state);
    float3 P{static_cast<float>(distr_range(curand_uniform(&state), -1.5, 0.3)),
             static_cast<float>(distr_range(curand_uniform(&state), -0.2, 0.4)),
             static_cast<float>(distr_range(curand_uniform(&state), -0.7, 0.5))};
    if (inside_tetrahedron(A, B, C, D, P))
        s_count[tid] = 1;
    else
        s_count[tid] = 0;

    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            s_count[tid] += s_count[tid + s];
        }
        __syncthreads();
    }
    // аггрегация на уровне сетки - atomicAdd
    if (tid == 0)
    {
        atomicAdd(accumulator, s_count[0]);
    }
}

__global__ void block_agregation_kernel(
    unsigned long long *global_mem,
    unsigned long long *result)
{
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
    cudaMalloc(&global_mem, sizeof(unsigned int));
    cudaMemset(global_mem, 0, sizeof(unsigned int));

    float3 *dev_vertices;
    cudaMalloc(&dev_vertices, sizeof(float3) * 4);
    cudaMemcpy(dev_vertices, vertices.data(), sizeof(float3) * 4, cudaMemcpyHostToDevice);

    volume_tetrahedron_on_device<<<grid_size, block_size, block_size * sizeof(unsigned int)>>>(
        dev_vertices[0],
        dev_vertices[1],
        dev_vertices[2],
        dev_vertices[3],
        N,
        accumulator,
        global_mem);

    //cudaDeviceSynchronize();
    cudaMemcpy(&count, accumulator, sizeof(int), cudaMemcpyDeviceToHost);

    return count;
}