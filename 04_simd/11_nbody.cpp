#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
    const int N = 8; // 配列サイズ
    float x[N], y[N], m[N], fx[N], fy[N];
    for (int i = 0; i < N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
        fx[i] = fy[i] = 0;
    }

    for (int i = 0; i < N; i++) {
        __m256 xi = _mm256_set1_ps(x[i]); // xiをベクトル化
        __m256 yi = _mm256_set1_ps(y[i]); // yiをベクトル化
        __m256 fxi = _mm256_setzero_ps(); // fx累積ベクトル
        __m256 fyi = _mm256_setzero_ps(); // fy累積ベクトル

        for (int j = 0; j < N; j += 8) { // 8粒子を一度に処理
            __m256 xj = _mm256_loadu_ps(&x[j]); // x[j]をベクトルでロード
            __m256 yj = _mm256_loadu_ps(&y[j]); // y[j]をベクトルでロード
            __m256 mj = _mm256_loadu_ps(&m[j]); // m[j]をベクトルでロード

            __m256 rx = _mm256_sub_ps(xi, xj); // rx = xi - xj
            __m256 ry = _mm256_sub_ps(yi, yj); // ry = yi - yj
            __m256 r2 = _mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry)); // r^2 = rx^2 + ry^2

            // 距離がゼロの場合を除外するためのマスク
            __m256 zero = _mm256_set1_ps(1e-15f);
            __m256 r2_mask = _mm256_cmp_ps(r2, zero, _CMP_GT_OQ); // r^2 > 1e-15

            // 距離の逆三乗を計算
            __m256 inv_r = _mm256_blendv_ps(_mm256_setzero_ps(), _mm256_rsqrt_ps(r2), r2_mask); // 1/sqrt(r^2)
            __m256 inv_r3 = _mm256_mul_ps(inv_r, _mm256_mul_ps(inv_r, inv_r)); // 1/r^3 = (1/sqrt(r^2))^3

            // 力の計算（マスク適用）
            __m256 fxi_term = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), rx), _mm256_mul_ps(mj, inv_r3)); // fx成分
            __m256 fyi_term = _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), ry), _mm256_mul_ps(mj, inv_r3)); // fy成分
            fxi = _mm256_add_ps(fxi, fxi_term); // fx累積
            fyi = _mm256_add_ps(fyi, fyi_term); // fy累積
        }

        // スカラー値に戻して格納
        float fx_temp[8], fy_temp[8];
        _mm256_storeu_ps(fx_temp, fxi);
        _mm256_storeu_ps(fy_temp, fyi);
        for (int k = 0; k < 8; k++) {
            fx[i] += fx_temp[k];
            fy[i] += fy_temp[k];
        }

        printf("%d %g %g\n", i, fx[i], fy[i]); // 結果出力
    }

    return 0;
}
