#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void count_keys(int *key, int *bucket, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicAdd(&bucket[key[idx]], 1);
    }
}

__global__ void sort_keys(int *key, int *bucket, int range) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < range) {
        int count = bucket[idx];
        for (int j = 0; j < count; j++) {
            key[idx * range + j] = idx;
        }
    }
}

int main() {
    int n = 50;
    int range = 5;
    std::vector<int> key(n);
    for (int i = 0; i < n; i++) {
        key[i] = rand() % range;
        printf("%d ", key[i]);
    }
    printf("\n");

    int *d_key, *d_bucket;
    cudaMalloc(&d_key, n * sizeof(int));
    cudaMalloc(&d_bucket, range * sizeof(int));
    cudaMemset(d_bucket, 0, range * sizeof(int));
    cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    count_keys<<<(n + 255) / 256, 256>>>(d_key, d_bucket, n);
    cudaDeviceSynchronize();

    sort_keys<<<(range + 255) / 256, 256>>>(d_key, d_bucket, range);
    cudaDeviceSynchronize();

    cudaMemcpy(key.data(), d_key, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        printf("%d ", key[i]);
    }
    printf("\n");

    cudaFree(d_key);
    cudaFree(d_bucket);

    return 0;
}
