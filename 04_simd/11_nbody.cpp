#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
    const int N = 16; // 配列サイズ（AVX512対応: 16の倍数）
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

            // インデックスマスクを作成（i != j の計算のみを実施）
            __m512i indices = _mm512_set_epi32(
                j+15, j+14, j+13, j+12, j+11, j+10, j+9, j+8,
                j+7, j+6, j+5, j+4, j+3, j+2, j+1, j);
            __m512i i_mask = _mm512_set1_epi32(i); // iをベクトル化
            __mmask16 comparison_mask = _mm512_cmp_epi32_mask(indices, i_mask, _MM_CMPINT_NE); // i != j のマスク

            // 力の計算（i == jを除外）
            __m512 rx = _mm512_sub_ps(xi, xj); // rx = xi - xj
            __m512 ry = _mm512_sub_ps(yi, yj); // ry = yi - yj
            __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry)); // r^2 = rx^2 + ry^2

            // 距離の逆三乗を計算（マスク適用）
            __m512 inv_r = _mm512_maskz_rsqrt14_ps(comparison_mask, r2); // 1/sqrt(r^2) (i == jを除外)
            __m512 inv_r3 = _mm512_mul_ps(inv_r, _mm512_mul_ps(inv_r, inv_r)); // 1/r^3 = (1/sqrt(r^2))^3

            // マスクを使用して力の計算
            __m512 fxi_term = _mm512_mask_mul_ps(_mm512_setzero_ps(), comparison_mask, rx, _mm512_mul_ps(mj, inv_r3)); // fx成分
            __m512 fyi_term = _mm512_mask_mul_ps(_mm512_setzero_ps(), comparison_mask, ry, _mm512_mul_ps(mj, inv_r3)); // fy成分
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
}
