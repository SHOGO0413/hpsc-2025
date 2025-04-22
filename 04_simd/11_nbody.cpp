#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
    const int N = 8;
    float x[N], y[N], m[N], fx[N], fy[N];

    for (int i = 0; i < N; i++) {
        x[i] = drand48();
        y[i] = drand48();
        m[i] = drand48();
    }

    //元々jループ内で使う予定だった要素をベクトル化
    __m512 xjvec = _mm512_load_ps(x);
    __m512 yjvec = _mm512_load_ps(y);
    __m512 mvec = _mm512_load_ps(m);

    for (int i = 0; i < N; i++) {

        //各xiとyiからxj,yjを引くことでループを除外
        __m512 xivec = _mm512_set1_ps(x[i]);
        __m512 yivec = _mm512_set1_ps(y[i]);

        __m512 rx = _mm512_sub_ps(xivec, xjvec); // rx = xi - xj
        __m512 ry = _mm512_sub_ps(yivec, yjvec); // ry = yi - yj

        //r2が0の場合および極端に小さい場合分母計算でエラーが出る可能性があるので除外
        __m512 r2 = _mm512_fmadd_ps(rx, rx, _mm512_mul_ps(ry, ry)); 
        __m512 safe_r2 = _mm512_max_ps(r2, _mm512_set1_ps(1e-5)); // 極端に小さい値を除外
        __m512 r_inv = _mm512_rsqrt14_ps(safe_r2);
        __m512 r3_inv = _mm512_mul_ps(r_inv, _mm512_mul_ps(r_inv, r_inv));

        //iループ目にi番目の要素(i=jに該当)の計算をしないように除外するマスク
        __m512i vi = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        __m512i vj = _mm512_set1_epi32(i);
        __mmask16 mask = _mm512_cmpneq_epi32_mask(vi, vj);  // i ≠ j のみ有効


        __m512 fxvec = _mm512_mask_mul_ps(_mm512_setzero_ps(), mask, rx, _mm512_mul_ps(mvec, r3_inv));
        __m512 fyvec = _mm512_mask_mul_ps(_mm512_setzero_ps(), mask, ry, _mm512_mul_ps(mvec, r3_inv));

        fx[i] = -_mm512_reduce_add_ps(fxvec);
        fy[i] = -_mm512_reduce_add_ps(fyvec);

        printf("%d %g %g\n", i, fx[i], fy[i]);

     }
 
     return 0;
}
