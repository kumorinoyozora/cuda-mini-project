#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


double current_time_ms() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec * 1000.0 + t.tv_usec / 1000.0;
}

void timeitC(double *matrix, size_t n, void (*testFunc) (double *, size_t)) {
    double start = current_time_ms();

    testFunc(matrix, n);

    double end = current_time_ms();
    printf("C time: %.3f ms\n\n", end - start);
}

void testGaussDetC(double *matrix, size_t n) {
    double det = determinantGauss(matrix, n);
    printf("Det (Gauss): %llf\n", det);
}

void testGaussInvC(double *matrix, size_t n) {
    double *revM = inversedFromGauss(matrix, n);
    if (!revM) {
        printf("det = 0 or idk\n");
        free(matrix);
        return;
    }
    printf("Inverted matrix (Gauss):\n");
    //printMatrix(revM, n);

    free(revM);
}

void testLUDetC(double *matrix, size_t n) {
    double *l = initL(n);
    double *u = allocateMatrix(n);

    LUdecomposition(matrix, l, u, n);

    double det = determinantLU(u, n);
    printf("Det (LU): %llf\n", det);
    
    free(l);
    free(u);
}

void testLUInvC(double *matrix, size_t n) {
    double *l = initL(n);
    double *u = allocateMatrix(n);

    LUdecomposition(matrix, l, u, n);

    double *inverted = inversedFromLU(l, u, n);
    printf("Inverted matrix (LU):\n");
    //printMatrix(inverted, n);
    
    free(l);
    free(u);
    free(inverted);
}

int main() {
    size_t n;
    double *matrix;
    inputMatrix(&matrix, &n);

    //
    //timeitC(matrix, n, testGaussDetC);
    //timeitC(matrix, n, testGaussInvC);
    timeitC(matrix, n, testLUDetC);
    timeitC(matrix, n, testLUInvC);
    //

    free(matrix); 
    return 0;
}