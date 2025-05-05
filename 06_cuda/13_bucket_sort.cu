#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void count_key(int* key, int* bucket, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&bucket[key[idx]], 1);
    }
}

void prefix_sum_cpu(int* bucket, int* prefix, int range) {
    prefix[0] = 0;
    for (int i = 1; i < range; i++) {
        prefix[i] = prefix[i - 1] + bucket[i - 1];
    }
}

__global__ void sort_key(int* sorted_key, int* prefix, int* bucket, int range) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < range && bucket[idx] > 0) {
        int start = prefix[idx];
        for (int i = 0; i < bucket[idx]; i++) {
            sorted_key[start + i] = idx;
        }
    }
}

int main() {
    int n = 50;
    int range = 5;

    // 初期データの準備
    std::vector<int> key(n);
    for (int i = 0; i < n; i++) {
        key[i] = rand() % range;
        printf("%d ", key[i]);
    }
    printf("\n");

    // GPUメモリ確保
    int *d_key, *d_bucket, *d_sorted;
    cudaMalloc(&d_key, n * sizeof(int));
    cudaMalloc(&d_bucket, range * sizeof(int));
    cudaMalloc(&d_sorted, n * sizeof(int));
    cudaMemset(d_bucket, 0, range * sizeof(int));

    cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // カウントカーネル呼び出し（名前変更）
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    count_key<<<blocks, threads>>>(d_key, d_bucket, n);
    cudaDeviceSynchronize();

    // カウント結果をCPUにコピー
    std::vector<int> bucket(range);
    cudaMemcpy(bucket.data(), d_bucket, range * sizeof(int), cudaMemcpyDeviceToHost);

    // prefix sum（CPUで処理）
    std::vector<int> prefix(range);
    prefix_sum_cpu(bucket.data(), prefix.data(), range);

    // prefixとbucketをGPUに送る
    int *d_prefix;
    cudaMalloc(&d_prefix, range * sizeof(int));
    cudaMemcpy(d_prefix, prefix.data(), range * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bucket, bucket.data(), range * sizeof(int), cudaMemcpyHostToDevice);

    // 再構築カーネル呼び出し（名前変更）
    sort_key<<<1, range>>>(d_sorted, d_prefix, d_bucket, range);
    cudaDeviceSynchronize();

    // 結果をCPUにコピー
    cudaMemcpy(key.data(), d_sorted, n * sizeof(int), cudaMemcpyDeviceToHost);

    // 結果表示
    for (int i = 0; i < n; i++) {
        printf("%d ", key[i]);
    }
    printf("\n");

    // メモリ解放
    cudaFree(d_key);
    cudaFree(d_bucket);
    cudaFree(d_sorted);
    cudaFree(d_prefix);

    return 0;
}
