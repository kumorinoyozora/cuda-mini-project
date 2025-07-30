#include "cumatrix.h"
#include <stdio.h>


void timeitCU(double *matrix, size_t n, void (*testFunc) (double *, size_t)) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    testFunc(matrix, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA time: %.3f ms\n\n", milliseconds);
}

void testGaussDetCU(double *matrix, size_t n) {
    double det = determinantGauss(matrix, n);
    printf("Det (Gauss): %llf\n", det);
}

void testGaussInvCU(double *matrix, size_t n) {
    double *revM = inversedFromGauss(matrix, n);
    if (!revM) {
        printf("det = 0 or idk\n");
        cudaFree(matrix);
        return;
    }
    printf("Inverted matrix (Gauss):\n");
    //printMatrix(revM, n);
    
    checkCuda( cudaFree(revM) );
}

void testLUDetCU(double *matrix, size_t n) {
    double *l = allocateMatrix(n);
    double *u = allocateMatrix(n);

    initL<<<getGridDim(n), THREADS_PER_BLOCK>>>(l, n);
    cudaDeviceSynchronize();

    LUdecomposition(matrix, l, u, n);

    double det = determinantLU(u, n);
    printf("Det (LU): %llf\n", det);
    
    checkCuda( cudaFree(l) ); 
    checkCuda( cudaFree(u) ); 
}

void testLUInvCU(double *matrix, size_t n) {
    double *l = allocateMatrix(n);
    double *u = allocateMatrix(n);

    initL<<<getGridDim(n), THREADS_PER_BLOCK>>>(l, n);
    cudaDeviceSynchronize();

    LUdecomposition(matrix, l, u, n);

    double *inversed = allocateMatrix(n);
    double *yAll = allocateMatrix(n);
    inversedFromLU<<<getGridDim(n), THREADS_PER_BLOCK>>>(l, u, inversed, yAll, n);
    cudaDeviceSynchronize();

    printf("Inverted matrix (LU):\n");
    //printMatrix(inverted, n);
    
    checkCuda( cudaFree(l) );
    checkCuda( cudaFree(u) );
    checkCuda( cudaFree(yAll) );
    checkCuda( cudaFree(inversed) );
}

int main() {
    size_t n;
    double *matrix;
    inputMatrix(&matrix, &n);

    //
    //timeitCU(matrix, n, testGaussDetCU); 
    //timeitCU(matrix, n, testGaussInvCU); 
    timeitCU(matrix, n, testLUDetCU); 
    timeitCU(matrix, n, testLUInvCU); 
    //

    cudaFree(matrix); 
    return 0;
}