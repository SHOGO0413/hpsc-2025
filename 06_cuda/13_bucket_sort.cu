#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

//各要素の出現回数を計算(keyの要素の出現回数をbucket配列に格納)
__global__ void count_key(int* key, int* bucket, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&bucket[key[idx]], 1);
    }
}

// 各バケットの値を並べ替えてsorted_keyに格納
__global__ void sort_key(int* sorted_key, int* bucket, int range, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < range) {
        int start = 0;
        // 各バケットの開始位置を計算
        for (int i = 0; i < idx; i++) {
            start += bucket[i]; 
        }
        // このバケットに含まれる値をsorted_keyに配置
        for (int i = 0; i < bucket[idx]; i++) {
            if (start + i < n) {
                sorted_key[start + i] = idx;
            }
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

    int *d_key, *d_bucket, *d_sorted;
    cudaMalloc(&d_key, n * sizeof(int));
    cudaMalloc(&d_bucket, range * sizeof(int));
    cudaMalloc(&d_sorted, n * sizeof(int));
    cudaMemset(d_bucket, 0, range * sizeof(int));

    cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // カウントカーネル呼び出し
    int threads = 1024;
    int blocks = (n + threads - 1) / threads;
    count_key<<<blocks, threads>>>(d_key, d_bucket, n);
    cudaDeviceSynchronize();

    // バケットソートカーネル呼び出し
    sort_key<<<1, range>>>(d_sorted, d_bucket, range, n);
    cudaDeviceSynchronize();

    // 結果をCPUにコピー
    cudaMemcpy(key.data(), d_sorted, n * sizeof(int), cudaMemcpyDeviceToHost);

    // 結果表示
    for (int i = 0; i < n; i++) {
        printf("%d ", key[i]);
    }
    printf("\n");

    cudaFree(d_key);
    cudaFree(d_bucket);
    cudaFree(d_sorted);
    return 0;
}
