#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>


/**
 * In fact, returns zero matrix
 */
double *allocateMatrix(size_t n) {
    double *matrix = (double *)calloc(n * n, sizeof(double));
    if (!matrix)
        return NULL;

    return matrix;
}

double *copyMatrix(double *a, size_t n) {
    if (!a)
        return NULL;

    double *b = allocateMatrix(n);
    if (!b)
        return NULL;

    memcpy(b, a, sizeof(double) * n * n);

    return b;
}

void inputMatrix(double **matrix, size_t *n) {
    scanf("%zu", n);
    *matrix = allocateMatrix(*n);

    for (size_t i = 0; i < *n * *n; i++) {
        scanf("%lf ", &(*matrix)[i]);
    }
}

void printMatrix(double *matrix, size_t n) {
    if (!matrix)
        return;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            printf("%.3lf ", matrix[i * n + j]);
        }
        putchar('\n');
    }
    putchar('\n');
}

double *transposed(double *a, size_t n) {
    if (!a) return NULL;

    double *t = allocateMatrix(n);
    if (!t) return NULL;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            t[j * n + i] = a[i * n + j];
        }
    }

    return t;
}

void multiplyByNum(double *a, size_t n, double num) {
    if (!a)
        return;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            a[i * n + j] *= num;
        }
    }
}

void swap(double *a, size_t n, size_t i_1, size_t i_2) {
    double temp;
    for (size_t j = 0; j < n; j++) {
        temp = a[i_1 * n + j];
        a[i_1 * n + j] = a[i_2 * n + j];
        a[i_2 * n + j] = temp;
    }
}

double determinantGauss(double *matrix, size_t n) {
    if (!matrix) return 0;

    double *a = copyMatrix(matrix, n);

    double det = 1;
    const double eps = 1e-9;
    for (size_t j = 0; j < n; j++) {
        size_t k = j;
        for (size_t i = j + 1; i < n; i++) {
            if (fabs(a[i * n + j]) > fabs(a[k * n + j])) k = i;
        }

        if (fabs(a[k * n + j]) < eps) return 0;

        if (j != k) {
            swap(a, n, j, k);
            det = -det;
        }
        det *= a[j * n + j];

        for (size_t i = j + 1; i < n; i++) {
            a[j * n + i] /= a[j * n + j];
        }

        for (size_t i = 0; i < n; i++) {
            if (j != i && fabs(a[i * n + j]) > eps) {
                for (size_t l = j + 1; l < n; l++) {
                    a[i * n + l] -= a[j * n + l] * a[i * n + j];
                }
            }
        }
    }

    free(a);
    return det;
}

double minor(double *a, size_t n, size_t i, size_t j) {
    if (!a) return 0;

    double *m = allocateMatrix(n - 1);
    if (!m) return 0;

    for (size_t row = 0, k = 0; row < n; row++) {
        if (row == i) continue;

        for (size_t col = 0, l = 0; col < n; col++) {
            if (col == j) continue;
            m[k * (n - 1) + l] = a[row * n + col];
            l++;
        }
        k++;
    }

    double mnr = determinantGauss(m, n - 1);
    free(m);

    return mnr;
}

double *algebraicAdditions(double *a, size_t n) {
    if (!a) return NULL;

    double *c = allocateMatrix(n);
    if (!c) return 0;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double M = minor(a, n, i, j);
            c[i * n + j] = ((i + j) % 2 == 0 ? 1.0 : -1.0) * M;
        }
    }

    return c;
}

double *inversedFromGauss(double *a, size_t n) {
    const double eps = 1e-9;
    double det = determinantGauss(a, n);
    if (fabs(det) < eps) return NULL;
    double revDet = 1 / det;

    double *algAdd = algebraicAdditions(a, n);

    double *algAddT = transposed(algAdd, n);

    multiplyByNum(algAddT, n, revDet);

    free(algAdd);

    return algAddT;
}

double *initL(size_t n) {
    double *l = allocateMatrix(n);
    if (!l) return NULL;

    for (size_t i = 0; i < n ; i++) {
        l[i * n + i] = 1.0;
    }

    return l;
}

void LUdecomposition(double *a, double *l, double *u, size_t n) {
    for (size_t i = 0; i < n; i++) {
        // U-строка
        for (size_t j = i; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < i; k++) {
                sum += l[i * n + k] * u[k * n + j];
            }
            u[i * n + j] = a[i * n + j] - sum;
        }

        // L-столбец
        for (size_t j = i; j < n; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < i; k++) {
                sum += l[j * n + k] * u[k * n + i];
            }
            l[j * n + i] = (a[j * n + i] - sum) / u[i * n + i];
        }
    }
}

double determinantLU(double *u, size_t n) {
    double det = 1.0;

    for (size_t i = 0; i < n; i++) {
        det *= u[i * n + i];
    }
    return det;
}

double *inversedFromLU(double *l, double *u, size_t n) {
    double *inv = allocateMatrix(n);

    // цикл по столбцам
    for (size_t j = 0; j < n; j++) {
        double *y = (double *)malloc(n * sizeof(double));

        // прямой ход L y = e_j
        for (size_t i = 0; i < n; i++) {
            y[i] = (i == j) ? 1.0 : 0.0;
            for (size_t k = 0; k < i; k++) {
                y[i] -= l[i * n + k] * y[k];
            }
        }

        // Обратный ход: U x = y
        for (size_t _i = 0; _i < n; _i++) { // for (int i = n - 1; i >= 0; i--)
            size_t i = n - 1 - _i;

            inv[i * n + j] = y[i];
            for (size_t k = i + 1; k < n; k++) {
                inv[i * n + j] -= u[i * n + k] * inv[k * n + j];
            }
            inv[i * n + j] /= u[i * n + i];
        }
        free(y);
    }

    return inv;
}