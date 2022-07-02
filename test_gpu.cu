#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>


void benchmark(int n, int size, int operand_gpu, int result_gpu) {
    // Creating vectors
    cudaSetDevice(0);
    thrust::device_vector<float> a(size, 1);

    cudaSetDevice(operand_gpu);
    thrust::device_vector<float> b(size, 2);

    cudaSetDevice(result_gpu);
    thrust::device_vector<float> c(size);

    // Timing objects
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    thrust::device_vector<float> times (0);

    // Benchmark
    for (int i=0; i<n; ++i) {
        cudaSetDevice(0);
        cudaEventRecord(start);
        thrust::transform(thrust::device, a.begin(), a.end(), b.begin(), c.begin(), thrust::plus<float>());
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        times.push_back(milliseconds);
    }

    float min = *thrust::min_element(times.begin(), times.end());
    float max = *thrust::max_element(times.begin(), times.end());
    float mean = thrust::reduce(times.begin(), times.end()) / float(n);
    printf("<d0> + <d%d> = <d%d> : n=%d min=%fms max=%fms mean=%f", operand_gpu, result_gpu, n, min, max, mean);
}


int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;

    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
    }

    if (deviceCount >= 2) {
        // Iterating over GPU
        for (unsigned device=0; device<deviceCount; ++device) {
            printf("Device %d access : ", device);
            for (unsigned other=0; other<deviceCount; ++other) {
                // Checking access
                int access = 0;
                cudaDeviceCanAccessPeer(&access, device, other);
                printf(" %d ", access);

                // Enabling peer access
                cudaDeviceEnablePeerAccess(device, other);
            }
            printf("\n");
        }
    }

    printf("Benchmark\n");

    unsigned int n = 1e5;
    unsigned int size = 1e6;
    for (unsigned int i=0; i<deviceCount; ++i) {
        benchmark(n, size, i, 0);
    }
}