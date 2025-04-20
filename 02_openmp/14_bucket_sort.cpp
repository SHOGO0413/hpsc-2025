#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main() {
  int n = 20;
  int range = 5;
  int tmp[range];
  std::vector<int> key(n);

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0);

  for (int i=0; i<n; i++)
    bucket[key[i]]++;
  std::vector<int> offset(range,0);

  #pragma omp parallel
  for (int j =1; j<range; j<<=1){
    #pragma omp parallel for
    for(int i = 0; i<range; i++){
        tmp[i]=offset[i];
    }
    #pragma omp parallel for
    for (int i=j; i<range; i++)
        offset[i] = tmp[i-1]+bucket[i-1];
  }

  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
