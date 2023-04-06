#include <iostream>
#include <cuda_runtime.h>

using namespace std;
int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cout << "Number of CUDA devices: " << deviceCount << endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        cout << "Device " << i << ": " << prop.name << endl;
        cout << "  Compute capability: " << prop.major << "." << prop.minor << endl;
        cout << "  Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << endl;
        cout << "  Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << endl;
        cout << "  Registers per block: " << prop.regsPerBlock << endl;
        cout << "  Warp size: " << prop.warpSize << endl;
        cout << "  Maximum threads per block: " << prop.maxThreadsPerBlock << endl;
        cout << "  Maximum thread dimensions: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << endl;
        cout << "  Maximum grid dimensions: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << endl;
    }

    return 0;
}
