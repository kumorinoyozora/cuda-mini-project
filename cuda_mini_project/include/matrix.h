#ifndef MATRIX_h
#define MATRIX_h


#include <stddef.h>

double *allocateMatrix(size_t n);
double *copyMatrix(double *a, size_t n);

void inputMatrix(double **matrix, size_t *n);
void printMatrix(double *matrix, size_t n);

double *transposed(double *a, size_t n);
void multiplyByNum(double *a, size_t n, double num);
void swap(double *a, size_t n, size_t i_1, size_t i_2);

double determinantGauss(double *matrix, size_t n);
double minor(double *a, size_t n, size_t i, size_t j);
double *algebraicAdditions(double *a, size_t n);
double *inversedFromGauss(double *a, size_t n);

double *initL(size_t n);
void LUdecomposition(double *a, double *l, double *u, size_t n);
double determinantLU(double *u, size_t n);
double *inversedFromLU(double *l, double *u, size_t n);

#endif