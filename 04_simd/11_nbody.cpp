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
    __m512 xjvec = _mm512_load_ps(x);
    __m512 yjvec = _mm512_load_ps(y);
    __m512 mvec = _mm512_load_ps(m);

    for (int i = 0; i < N; i++) {
        
        __m512 xivec = _mm512_set1_ps(x[i]); // xiをベクトル化
        __m512 yivec = _mm512_set1_ps(y[i]); // yiをベクトル化

        __m512 fxvec = _mm512_setzero_ps();
        __m512 fyvec = _mm512_setzero_ps();
        
        __m512 rx = _mm512_sub_ps(xivec, xjvec); // rx = xi - xj
        __m512 ry = _mm512_sub_ps(yivec, yjvec); // ry = yi - yj
        __m512 r2 = _mm512_fmadd_ps(rx, rx, _mm512_mul_ps(ry, ry)); 
        __m512 safe_r2 = _mm512_max_ps(r2, _mm512_set1_ps(1e-5)); // 極端に小さい値を除外
        __m512 r_inv = _mm512_rsqrt14_ps(safe_r2);
        __m512 r3_inv = _mm512_mul_ps(r_inv, _mm512_mul_ps(r_inv, r_inv));

        __m512i vi = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m512i vj = _mm512_set1_epi32(i);
        __mmask16 mask = _mm512_cmpneq_epi32_mask(vi, vj);  // i ≠ j のみ有効


        __m512 fxi_term = _mm512_mask_mul_ps(_mm512_setzero_ps(), mask, rx, _mm512_mul_ps(mvec, r3_inv)); // fx成分
        __m512 fyi_term = _mm512_mask_mul_ps(_mm512_setzero_ps(), mask, ry, _mm512_mul_ps(mvec, r3_inv)); // fy成分

        fxvec = _mm512_sub_ps(fxvec, fxi_term); // fx累積
        fyvec = _mm512_sub_ps(fyvec, fyi_term); // fy累積

        fx[i] = _mm512_reduce_add_ps(fxi_term);
        fy[i] = _mm512_reduce_add_ps(fyi_term);

        printf("%d %g %g\n", i, fx[i], fy[i]);

     }
 
     return 0;
}
