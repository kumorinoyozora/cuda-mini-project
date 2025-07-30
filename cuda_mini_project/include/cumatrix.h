#ifndef CU_MATRIX_h
#define CU_MATRIX_h


#include <stddef.h>
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>


#define THREADS_PER_BLOCK 256
#define checkCuda(val) checkCudaImpl((val), #val, __FILE__, __LINE__)

/**
 * for kernels check: checkCuda( cudaGetLastError() );
 * for cuda functions check: checkCuda( cudaDeviceSynchronize );
 */
__host__ inline cudaError_t checkCudaImpl(cudaError_t result, const char *func, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error at: %s: line %d in %s: %s\n", 
            file, line, func, cudaGetErrorString(result));
        assert(false);
    }
    return result;
}

inline dim3 getGridDim(size_t totalWork, size_t threadsPerBlock = 256) {
    return dim3((totalWork - 1) / threadsPerBlock + 1);
}

double *allocateMatrix(size_t n);
double *copyMatrix(double *a, size_t n);

void inputMatrix(double **matrix, size_t *n);
void printMatrix(double *matrix, size_t n);

__global__ void transpose(double *matrix, size_t n);
__global__ void multiplyByNum(double *a, size_t n, double num);
__global__ void swap(double *a, size_t n, size_t i_1, size_t i_2);

__global__ void findPivotIndex(double *a, size_t n, size_t j, size_t *result);
__global__ void normalizePivot(double *a, size_t n, size_t j);
__global__ void eliminateBelow(double *a, size_t n, size_t j, const double eps);
double determinantGauss(double *matrix, size_t n);
double minor(double *a, size_t n, size_t i, size_t j);
double *minorsMatrix(double *a, size_t n);
__global__ void algebraicAdditions(double *a, double *m, double *c, size_t n);
double *inversedFromGauss(double *a, size_t n);

__global__ void initL(double *l, size_t n);
__global__ void updateLU(double *a, double *l, double *u, size_t n, size_t i);
void LUdecomposition(double *a, double *l, double *u, size_t n);
double determinantLU(double *u, size_t n);
__global__ void inversedFromLU(double *l, double *u, double *inv, double *yAll, size_t n);


#endif