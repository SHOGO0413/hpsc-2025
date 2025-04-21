#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
    const int N = 16; // 配列サイズ
    float x[N], y[N], m[N], fx[N], fy[N];
    for (int i = 0; i < N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
        fx[i] = fy[i] = 0;
    }

    for (int i = 0; i < N; i++) {
        __m512 xi = _mm512_set1_ps(x[i]); // xiをベクトル化
        __m512 yi = _mm512_set1_ps(y[i]); // yiをベクトル化
        __m512 fxi = _mm512_setzero_ps(); // fx累積ベクトル
        __m512 fyi = _mm512_setzero_ps(); // fy累積ベクトル

        for (int j = 0; j < N; j += 16) { // 16粒子を一度に処理
            __m512 xj = _mm512_loadu_ps(&x[j]); // x[j]をベクトルでロード
            __m512 yj = _mm512_loadu_ps(&y[j]); // y[j]をベクトルでロード
            __m512 mj = _mm512_loadu_ps(&m[j]); // m[j]をベクトルでロード

            // 距離計算
            __m512 rx = _mm512_sub_ps(xi, xj); // rx = xi - xj
            __m512 ry = _mm512_sub_ps(yi, yj); // ry = yi - yj
            __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry)); // r^2 = rx^2 + ry^2

            // 距離がゼロの場合を除外するためのマスク
            __m512 zero = _mm512_set1_ps(1e-15f);
            __mmask16 r2_mask = _mm512_cmp_ps_mask(r2, zero, _CMP_GT_OQ); // r^2 > 1e-15

            // 距離の逆三乗を計算（マスク適用）
            __m512 inv_r = _mm512_maskz_rsqrt14_ps(r2_mask, r2); // 1/sqrt(r^2) (距離ゼロを除外)
            __m512 inv_r3 = _mm512_mul_ps(inv_r, _mm512_mul_ps(inv_r, inv_r)); // 1/r^3 = (1/sqrt(r^2))^3

            // 力の計算（マスク適用）
            __m512 fxi_term = _mm512_mask_mul_ps(_mm512_setzero_ps(), r2_mask, rx, _mm512_mul_ps(mj, inv_r3)); // fx成分
            __m512 fyi_term = _mm512_mask_mul_ps(_mm512_setzero_ps(), r2_mask, ry, _mm512_mul_ps(mj, inv_r3)); // fy成分
            fxi = _mm512_add_ps(fxi, fxi_term); // fx累積
            fyi = _mm512_add_ps(fyi, fyi_term); // fy累積
        }

        // スカラー値に戻して格納
        float fx_temp[16], fy_temp[16];
        _mm512_storeu_ps(fx_temp, fxi);
        _mm512_storeu_ps(fy_temp, fyi);
        for (int k = 0; k < 16; k++) {
            fx[i] += fx_temp[k];
            fy[i] += fy_temp[k];
        }

        printf("%d %g %g\n", i, fx[i], fy[i]); // 結果出力
    }

    return 0;
}
