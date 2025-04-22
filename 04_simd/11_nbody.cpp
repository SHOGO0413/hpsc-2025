#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
    const int N = 8;
    float x[N], y[N], m[N], fx[N], fy[N];

    // 初期化
    for (int i = 0; i < N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
    }

    // ベクトル化用のAVX-512レジスタ
    __m512 vx = _mm512_loadu_ps(x);
    __m512 vy = _mm512_loadu_ps(y);
    __m512 vm = _mm512_loadu_ps(m);
    __m512 vfx = _mm512_setzero_ps();
    __m512 vfy = _mm512_setzero_ps();

    // SIMD演算（ペアワイズ計算）
    for (int j = 0; j < N; j++) {
        __m512 rx = _mm512_sub_ps(vx, _mm512_set1_ps(x[j]));
        __m512 ry = _mm512_sub_ps(vy, _mm512_set1_ps(y[j]));
        __m512 r2 = _mm512_fmadd_ps( _mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry));
        __m512 r_inv = _mm512_rsqrt14_ps(r2);
        __m512 r3_inv = _mm512_mul_ps(r_inv, _mm512_mul_ps(r_inv, r_inv));
        __m512 vMj = _mm512_set1_ps(m[j]);

        // 力の計算
        vfx = _mm512_fnmadd_ps(rx, _mm512_mul_ps(vMj, r3_inv), vfx);
        vfy = _mm512_fnmadd_ps(ry, _mm512_mul_ps(vMj, r3_inv), vfy);
    }

    // 結果を保存
    _mm512_storeu_ps(fx, vfx);
    _mm512_storeu_ps(fy, vfy);

    // 出力
    for (int i = 0; i < N; i++) {
        printf("%d %g %g\n", i, fx[i], fy[i]);
    }

    return 0;
}
