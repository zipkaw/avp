#include "device_fun.cu"
#include "host_fun.cu"
#include <iostream>
#include <random>
#include <chrono>
#include <array>

#define distr_range(random_fun, min, max) (random_fun * (max - (min))) + min
#define volume(V, Np, Ni) (V * (Np * (1.0 / Ni)))

using namespace std;

__host__ __device__ float tetrahedron_volume_analitic(const float3 &A,
                                                      const float3 &B,
                                                      const float3 &C,
                                                      const float3 &D)
{
    float x1 = A.x;
    float y1 = A.y;
    float z1 = A.z;

    float x2 = B.x;
    float y2 = B.y;
    float z2 = B.z;

    float x3 = C.x;
    float y3 = C.y;
    float z3 = C.z;

    float x4 = D.x;
    float y4 = D.y;
    float z4 = D.z;

    float term1 = (x2 - x1) * ((y3 - y1) * (z4 - z1) - (y4 - y1) * (z3 - z1));
    float term2 = (y2 - y1) * ((z3 - z1) * (x4 - x1) - (z4 - z1) * (x3 - x1));
    float term3 = (z2 - z1) * ((x3 - x1) * (y4 - y1) - (x4 - x1) * (y3 - y1));

    return std::abs((1.0f / 6.0f) * (term1 + term2 + term3));
}

int main()
{
    array<float3, 4> vertices = {
        float3{0, 0, -0.7},   // A
        float3{-1.5, 0, 0},   // B
        float3{0, -0.2, 0},   // C
        float3{0.3, 0.4, 0.5} // D
    };
    float vol = (1.8 * 0.6 * 1.2);

    int N = 1'000'000;
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
    chrono::high_resolution_clock::time_point t1 = chrono::high_resolution_clock::now();
    float HOSTvolume = volume_tetrahedron(vertices[0], vertices[1], vertices[2], vertices[3], N);
    chrono::high_resolution_clock::time_point t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> time_span = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    std::cout << "Volume of tetrahedron: " << HOSTvolume << std::endl;
    // std::cout << "Volume of tetrahedron: " << volume(vol, HOSTvolume, N) << std::endl;
    std::cout << "Time: " << time_span.count() << " ms" << std::endl;

    /*---------CUDA--------------*/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // cout << "Volume on device: " << volume(vol, device_estimate(vertices, N), N) << endl;
    cout << "Volume on device: " << device_estimate(vertices, N) << endl;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time: " << elapsedTime << " ms" << std::endl;
    /*---------TUDA--------------*/

    cout << "Volume analitic: " << tetrahedron_volume_analitic(vertices[0], vertices[1], vertices[2], vertices[3]) << endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}