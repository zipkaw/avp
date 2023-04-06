#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>
#include <chrono>
#include <array>
#include <eigen/Eigen/Dense>
#include <ctgmath>
// #include <mtl.h>
#define BLOCK_SIZE 256
#define distr_range(random_fun, min, max) (random_fun * (max - (min))) + min

using namespace Eigen;
using namespace std;

float pointA[] = {0.0, 0.0, -0.7};
float pointB[] = {-1.5, 0.0, 0.0};
float pointC[] = {0.0, -0.2, 0.0};
float pointD[] = {0.3, 0.4, 0.5};

float xmin = -1.5;
float xmax = 0.3;
float ymin = -0.2;
float ymax = 0.4;
float zmin = -0.7;
float zmax = 0.5;

float xyzrg[] = {-1.5,
                 0.3,
                 -0.2,
                 0.4,
                 -0.7,
                 0.5};

float3 Hcross(const float3 &a, const float3 &b)
{
    float3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

float Hdot(const float3 &a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
float3 Hnormalized(const float3 &a)
{
    float r = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    float3 norm_vector{a.x / r, a.y / r, a.z / r};
    return norm_vector;
}

__device__ float3 cross(const float3 &a, const float3 &b)
{
    float3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

__device__ float dot(const float3 &a, const float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 normalized(const float3 &a)
{
    float r = sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
    float3 norm_vector{a.x / r, a.y / r, a.z / r};
    return norm_vector;
}

// bool inside_tetrahedron(const Vector3d &A, const Vector3d &B, const Vector3d &C, const Vector3d &D, const Vector3d &P)
// {
//     Vector3d N1 = (B - A).cross(C - A).normalized();
//     Vector3d N2 = (C - B).cross(D - B).normalized();
//     Vector3d N3 = (D - C).cross(A - C).normalized();
//     Vector3d N4 = (A - D).cross(B - D).normalized();
//     float d1 = N1.dot(A);
//     float d2 = N2.dot(B);
//     float d3 = N3.dot(C);
//     float d4 = N4.dot(D);

//     float dist1 = N1.dot(P) - d1;
//     float dist2 = N2.dot(P) - d2;
//     float dist3 = N3.dot(P) - d3;
//     float dist4 = N4.dot(P) - d4;

//     if ((std::signbit(dist1) == std::signbit(N1.dot(D) - d1)) &&
//         (std::signbit(dist2) == std::signbit(N2.dot(C) - d2)) &&
//         (std::signbit(dist3) == std::signbit(N3.dot(A) - d3)) &&
//         (std::signbit(dist4) == std::signbit(N4.dot(B) - d4)))
//     {
//         return true;
//     }
//     else
//     {
//         return false;
//     }
// }

bool inside_tetrahedron(const float3 &A, const float3 &B, const float3 &C, const float3 &D, const float3 &P)
{
    float3 AB{B.x - A.x, B.y - A.y, B.z - A.z};
    float3 AC{C.x - A.x, C.y - A.y, C.z - A.z};
    float3 BC{C.x - B.x, C.y - B.y, C.z - B.z};
    float3 BD{D.x - B.x, D.y - B.y, D.z - B.z};
    float3 CD{D.x - C.x, D.y - C.y, D.z - C.z};
    float3 CA{A.x - C.x, A.y - C.y, A.z - C.z};
    float3 DA{A.x - D.x, A.y - D.y, A.z - D.z};
    float3 DB{B.x - D.x, B.y - D.y, B.z - D.z};

    float3 n1 = Hcross(AB, AC);
    n1 = Hnormalized(n1);
    float3 n2 = Hcross(BC, BD);
    n2 = Hnormalized(n2);
    float3 n3 = Hcross(CD, CA);
    n3 = Hnormalized(n3);
    float3 n4 = Hcross(DA, DB);
    n4 = Hnormalized(n4);

    float d1 = Hdot(n1, A);
    float d2 = Hdot(n2, B);
    float d3 = Hdot(n3, C);
    float d4 = Hdot(n4, D);

    float dist1 = Hdot(n1, P) - d1;
    float dist2 = Hdot(n2, P) - d2;
    float dist3 = Hdot(n3, P) - d3;
    float dist4 = Hdot(n4, P) - d4;

    if ((std::signbit(dist1) == std::signbit(Hdot(n1, D) - d1)) &&
        (std::signbit(dist2) == std::signbit(Hdot(n2, C) - d2)) &&
        (std::signbit(dist3) == std::signbit(Hdot(n3, A) - d3)) &&
        (std::signbit(dist4) == std::signbit(Hdot(n4, B) - d4)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

__device__ bool inside_tetrahedron_on_device(const float3 &A, const float3 &B, const float3 &C, const float3 &D, const float3 &P)
{
    float3 AB{B.x - A.x, B.y - A.y, B.z - A.z};
    float3 AC{C.x - A.x, C.y - A.y, C.z - A.z};
    float3 BC{C.x - B.x, C.y - B.y, C.z - B.z};
    float3 BD{D.x - B.x, D.y - B.y, D.z - B.z};
    float3 CD{D.x - C.x, D.y - C.y, D.z - C.z};
    float3 CA{A.x - C.x, A.y - C.y, A.z - C.z};
    float3 DA{A.x - D.x, A.y - D.y, A.z - D.z};
    float3 DB{B.x - D.x, B.y - D.y, B.z - D.z};

    float3 n1 = cross(AB, AC);
    n1 = normalized(n1);
    float3 n2 = cross(BC, BD);
    n2 = normalized(n2);
    float3 n3 = cross(CD, CA);
    n3 = normalized(n3);
    float3 n4 = cross(DA, DB);
    n4 = normalized(n4);

    float d1 = dot(n1, A);
    float d2 = dot(n2, B);
    float d3 = dot(n3, C);
    float d4 = dot(n4, D);

    float dist1 = dot(n1, P) - d1;
    float dist2 = dot(n2, P) - d2;
    float dist3 = dot(n3, P) - d3;
    float dist4 = dot(n4, P) - d4;

    if ((std::signbit(dist1) == std::signbit(dot(n1, D) - d1)) &&
        (std::signbit(dist2) == std::signbit(dot(n2, C) - d2)) &&
        (std::signbit(dist3) == std::signbit(dot(n3, A) - d3)) &&
        (std::signbit(dist4) == std::signbit(dot(n4, B) - d4)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

float volume_tetrahedron(const float3 &A, const float3 &B, const float3 &C, const float3 &D, int N)
{
    default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    uniform_real_distribution<float> distributionX(xmin, xmax),
        distributionY(ymin, ymax),
        distributionZ(zmin, zmax);

    int count = 0;
    for (int i = 0; i < N; i++)
    {
        float3 P{distributionX(generator),
                 distributionY(generator),
                 distributionZ(generator)};

        if (inside_tetrahedron(A, B, C, D, P))
        {
            count++;
        }
    }

    float volume_tetrahedron = ((xmax - xmin) *
                                (ymax - ymin) *
                                (zmax - zmin)) *
                               (count * (1.0 / N));
    return volume_tetrahedron;
}

__global__ void volume_tetrahedron_on_device(const float3 &A,
                                             const float3 &B,
                                             const float3 &C,
                                             const float3 &D,
                                             int N,
                                             float *accumulator,
                                             float *xyzrange)
{
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int num_blocks = gridDim.x;
    int num_points_per_block = (N + num_blocks - 1) / num_blocks;
    int start = num_points_per_block * blockIdx.x;
    int end = min(start + num_points_per_block, N);

    extern __shared__ float count[];

    volatile size_t local_count = 0;
    curandState state;
    curand_init(1234UL, tid, 0, &state);

    for (int i = start; i < end; i++)
    {
        float3 P{distr_range(curand_uniform(&state), -1.5, 0.3),
                 distr_range(curand_uniform(&state), -0.2, 0.4),
                 distr_range(curand_uniform(&state), -0.7, 0.5)};
        if (inside_tetrahedron_on_device(A, B, C, D, P))
        {
            local_count++;
        }
    }

    // atomicAdd(&count[tid], static_cast<float>(local_count));

    // for (unsigned int s = num_threads / 2; s > 0; s >>= 1)
    // {
    //     if (tid < s)
    //     {
    //         count[tid] += count[tid + s];
    //     }
    //     __syncthreads();
    // }

    // accumulator = &count[0];
}

int main()
{
    // array<Vector3d, 4> vertices = {
    //     Vector3d(0, 0, -0.7),   // A
    //     Vector3d(-1.5, 0, 0),   // B
    //     Vector3d(0, -0.2, 0),   // C
    //     Vector3d(0.3, 0.4, 0.5) // D
    // };
    array<float3, 4> vertices = {
        float3{0, 0, -0.7},   // A
        float3{-1.5, 0, 0},   // B
        float3{0, -0.2, 0},   // C
        float3{0.3, 0.4, 0.5} // D
    };
    array<float3, 4> vertices_dev = {
        float3{0, 0, -0.7},   // A
        float3{-1.5, 0, 0},   // B
        float3{0, -0.2, 0},   // C
        float3{0.3, 0.4, 0.5} // D
    };
    int N = 100'000;
    int N_dev = 10000;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Device " << 0 << ": " << prop.name << endl;
    cout << "  Compute capability: " << prop.major << "." << prop.minor << endl;
    cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
    cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
    cout << "  Registers per block: " << prop.regsPerBlock << endl;
    cout << "  Warp size: " << prop.warpSize << endl;
    cout << "  Maximum threads per block: " << prop.maxThreadsPerBlock << endl;
    cout << "  Maximum thread dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << endl;
    cout << "  Maximum grid dimensions: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << endl;

    float HOSTvolume = volume_tetrahedron(vertices[0], vertices[1], vertices[2], vertices[3], N);
    std::cout << "Volume of tetrahedron: " << HOSTvolume << std::endl;

    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    // dim3 blockDimensions{block_size};
    // dim3 gridDimensions{block_count};

    size_t shared_mem = BLOCK_SIZE * sizeof(float);
    float3 *dev_vertices;

    float *accumulator;
    cudaMalloc(&accumulator, sizeof(float));
    cudaMemset(accumulator, 0, sizeof(float));

    cudaMalloc(&dev_vertices, sizeof(float3) * 4);
    cudaMemcpy(dev_vertices, vertices.data(), sizeof(float3) * 4, cudaMemcpyHostToDevice);

    float *xyzrange;

    cudaMalloc(&xyzrange, sizeof(float));
    cudaMemset(xyzrange, 0, sizeof(float));
    cudaMemcpy(xyzrange, xyzrg, sizeof(float) * 6, cudaMemcpyHostToDevice);

    volume_tetrahedron_on_device<<<gridDim, blockDim, shared_mem>>>(vertices_dev[0],
                                                                                  vertices_dev[1],
                                                                                  vertices_dev[2],
                                                                                  vertices_dev[3],
                                                                                  N_dev,
                                                                                  accumulator,
                                                                                  xyzrange);

    float result;

    cudaMemcpy(&result, accumulator, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "volume on device" << result << endl;

    return 0;
}