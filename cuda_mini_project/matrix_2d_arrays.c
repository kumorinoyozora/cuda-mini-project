/**
 * BASELINE FILE
 * 
 * CUDA не работает с двумерными C массивами, поэтому исходное решение без параллелизма 
 * берётся из файла matrix_1d_2d_arrays.c
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>

/**
 * In fact, returns zero matrix
 */
double **allocateMatrix(size_t n) {
    double **matrix = (double **)malloc(n * sizeof(double *));
    if (!matrix) return NULL;

    for (size_t i = 0; i < n; i++) {
        matrix[i] = (double *)calloc(n, sizeof(double));
    }

    return matrix;
}

void freeMatrix(double **matrix, size_t n) {
    if (!matrix) return;

    for (size_t i = 0; i < n; i++) {
        free(matrix[i]);
    }

    free(matrix);
}

double **copyMatrix(double **a, size_t n) {
    if (!a) return NULL;

    double **b = allocateMatrix(n);
    if (!b) return NULL;

    for (size_t i = 0; i < n; i++) {
        memcpy(b[i], a[i], n * sizeof(double));
    }

    return b;
}

void printMatrix(double **matrix, size_t n) {
    if (!matrix) return;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            printf("%.3lf ", matrix[i][j]);
        }
        putchar('\n');
    }
    putchar('\n');
}

void multiplyByNum(double **a, size_t n, double num) {
    if (!a) return;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            a[i][j] *= num;
        }
    }
}

void swap(double **a, size_t i_1, size_t i_2) {
    double *temp = a[i_1];
    a[i_1] = a[i_2];
    a[i_2] = temp;
}

double determinant(double **matrix, size_t n) {
    if (!matrix) return 0;

    double **a = copyMatrix(matrix, n);

    double det = 1;
    const double eps = 1e-9;
    for (size_t j = 0; j < n; j++) {
        size_t k = j;
        for (size_t i = j + 1; i < n; i++) {
            if (fabs(a[i][j]) > fabs(a[k][j])) k = i;
        }

        if (fabs(a[k][j]) < eps) return 0;

        if (j != k) {
            swap(a, j, k);
            det = -det;
        }
        det *= a[j][j];

        for (size_t i = j + 1; i < n; i++) {
            a[j][i] /= a[j][j];
        }
        for (size_t i = 0; i < n; i++) {
            if (j != i && fabs(a[i][j]) > eps) {
                for (size_t l = j + 1; l < n; l++) {
                    a[i][l] -= a[j][l] * a[i][j];
                }
            }
        }
    }
    freeMatrix(a, n);
    return det;
}

double minor(double **a, size_t n, size_t i, size_t j) {
    if (!a) return 0;

    double **m = allocateMatrix(n - 1);
    if (!m) return 0;

    for (size_t row = 0, k = 0; row < n; row++) {
        if (row == i) continue;

        for (size_t col = 0, l = 0; col < n; col++) {
            if (col == j) continue;
            m[k][l] = a[row][col];
            l++;
        }
        k++;
    }

    double mnr = determinant(m, n - 1);
    freeMatrix(m, n - 1);

    return mnr;
}

double **transposed(double **a, size_t n) {
    if (!a) return NULL;

    double **t = allocateMatrix(n);
    if (!t) return NULL;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            t[j][i] = a[i][j];
        }
    }

    return t;
}

double **algebraicAdditions(double **a, size_t n) {
    if (!a) return NULL;

    double **c = allocateMatrix(n);
    if (!c) return 0;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            double M = minor(a, n, i, j);
            c[i][j] = ((i + j) % 2 == 0 ? 1.0 : -1.0) * M;
        }
    }

    return c;
}

double **reversedMatrix(double **a, size_t n) {
    const double eps = 1e-9;
    double det = determinant(a, n);
    if (fabs(det) < eps) return NULL;
    double revDet = 1 / det;

    double **algAdd = algebraicAdditions(a, n);

    double **algAddT = transposed(algAdd, n);

    multiplyByNum(algAddT, n, revDet);

    freeMatrix(algAdd, n);

    return algAddT;
}

void inputMatrix(double ***matrix, size_t *n) {
    scanf("%zu", n);
    *matrix = allocateMatrix(*n);

    for (size_t i = 0; i < *n; i++) {
        for (size_t j = 0; j < *n; j++) {
            scanf("%lf ", &(*matrix)[i][j]);
        }
    }
}

int main() {
    size_t n;
    double **matrix;
    inputMatrix(&matrix, &n);

    double **revM = reversedMatrix(matrix, n);
    if (!revM) {
        printf("det = 0 or idk\n");
        freeMatrix(matrix, n);
        return 0;
    }
    
    puts("Inverted matrix:\n");
    printMatrix(revM, n);
    
    freeMatrix(revM, n);
    freeMatrix(matrix, n); 

    return 0;
}

/* int main() {
    size_t n = 3;
    double **matrix = allocateMatrix(n);
    matrix[0][0] = 4.0;
    matrix[0][1] = 2.0;
    matrix[0][2] = 2.0;
    matrix[1][0] = 3.0;
    matrix[1][1] = 3.0;
    matrix[1][2] = 3.0;
    matrix[2][0] = 1.0;
    matrix[2][1] = 2.0;
    matrix[2][2] = 5.0;
    
    double **revM = reversedMatrix(matrix, n);
    if (!revM) {
        printf("det = 0 or idk\n");
        freeMatrix(matrix, n);
        return 0;
    }
    
    printMatrix(matrix, n);
    printMatrix(revM, n);

    freeMatrix(matrix, n);
    freeMatrix(revM, n);

    return 0;
} */