#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h> // AVX512ヘッダーをインクルード

int main() {
    const int N = 16; // 配列サイズ（AVX512は16要素に対応）
    float x[N], y[N], m[N], fx[N], fy[N];

    // ランダム初期化
    for (int i = 0; i < N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
        fx[i] = fy[i] = 0;
    }

    for (int i = 0; i < N; i++) {
        __m512 fx_vec = _mm512_setzero_ps(); // fxの累積ベクトル
        __m512 fy_vec = _mm512_setzero_ps(); // fyの累積ベクトル

        __m512 xi_vec = _mm512_set1_ps(x[i]); // xiをベクトル形式で設定
        __m512 yi_vec = _mm512_set1_ps(y[i]); // yiをベクトル形式で設定

        for (int j = 0; j < N; j += 16) { // AVX512は16要素を並列処理
            __m512 xj_vec = _mm512_loadu_ps(&x[j]); // x[j]のベクトルをロード
            __m512 yj_vec = _mm512_loadu_ps(&y[j]); // y[j]のベクトルをロード
            __m512 mj_vec = _mm512_loadu_ps(&m[j]); // m[j]のベクトルをロード

            __m512 rx_vec = _mm512_sub_ps(xi_vec, xj_vec); // rx = xi - xj
            __m512 ry_vec = _mm512_sub_ps(yi_vec, yj_vec); // ry = yi - yj
            __m512 r2_vec = _mm512_add_ps(_mm512_mul_ps(rx_vec, rx_vec), _mm512_mul_ps(ry_vec, ry_vec)); // r^2

            // 逆平方根の計算（rの逆平方根により計算を高速化）
            __m512 inv_r_vec = _mm512_rsqrt14_ps(r2_vec); // 1 / sqrt(r^2)
            __m512 inv_r3_vec = _mm512_mul_ps(inv_r_vec, _mm512_mul_ps(inv_r_vec, inv_r_vec)); // 1 / r^3

            // 力の計算
            __m512 fx_term = _mm512_mul_ps(rx_vec, _mm512_mul_ps(mj_vec, inv_r3_vec)); // fx項
            __m512 fy_term = _mm512_mul_ps(ry_vec, _mm512_mul_ps(mj_vec, inv_r3_vec)); // fy項
            fx_vec = _mm512_sub_ps(fx_vec, fx_term); // fx累積
            fy_vec = _mm512_sub_ps(fy_vec, fy_term); // fy累積
        }

        // fx, fyをスカラーに戻して格納
        float fx_temp[16], fy_temp[16];
        _mm512_storeu_ps(fx_temp, fx_vec);
        _mm512_storeu_ps(fy_temp, fy_vec);
        for (int k = 0; k < 16; k++) {
            fx[i] += fx_temp[k];
            fy[i] += fy_temp[k];
        }

        printf("%d %g %g\n", i, fx[i], fy[i]); // 結果出力
    }

    return 0;
}
