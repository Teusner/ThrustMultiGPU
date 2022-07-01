#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>


int main(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;

    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
    }

    if (deviceCount > 2) {
        unsigned int size = 1000;
        cudaSetDevice(0);
        thrust::device_vector<float> a(size, 2);
        thrust::device_vector<float> c(size);

        cudaSetDevice(1);
        thrust::device_vector<float> b(size, 1);
        thrust::device_vector<float> d(size);

        cudaSetDevice(0);
        thrust::transform(a.begin(), a.end(), b.begin(), c.begin(), thrust::plus<float>());

        cudaSetDevice(1);
        thrust::transform(a.begin(), a.end(), b.begin(), d.begin(), thrust::minus<float>());
    }
}