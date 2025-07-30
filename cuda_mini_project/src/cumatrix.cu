#include "cumatrix.h"
#include <stdlib.h>
#include <math.h>


/**
 * In fact, returns zero matrix
 */
double *allocateMatrix(size_t n) {
    double *matrix;
    checkCuda( cudaMallocManaged(&matrix, sizeof(double) * n * n) );
    if (!matrix) return NULL;

    return matrix;
}

double *copyMatrix(double *a, size_t n) {
    if (!a) return NULL;

    double *b = allocateMatrix(n);
    if (!b) return NULL;

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
    if (!matrix) return;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            printf("%.3lf ", matrix[i * n + j]);
        }
        putchar('\n');
    }
    putchar('\n');
}

__global__ void transpose(double *matrix, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t gridStride = gridDim.x * blockDim.x;

    for (size_t k = idx; k < n * n; k += gridStride) {
        size_t i = k / n;
        size_t j = k % n;

        if (i < j) { // чтобы не поменять обратно
            double temp = matrix[i * n + j];
            matrix[i * n + j] = matrix[j * n + i];
            matrix[j * n + i] = temp;
        }
    }
}

__global__ void multiplyByNum(double *a, size_t n, double num) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t gridStride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n * n; i += gridStride) {
        a[i] *= num;
    }
}

__global__ void swap(double *a, size_t n, size_t i_1, size_t i_2) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t gridStride = gridDim.x * blockDim.x;

    for (size_t j = idx; j < n; j += gridStride) {
        double temp = a[i_1 * n + j];
        a[i_1 * n + j] = a[i_2 * n + j];
        a[i_2 * n + j] = temp;
    }
}

__global__ void findPivotIndex(double *a, size_t n, size_t j, size_t *result) {
    extern __shared__ char shared[];
    double *sdata = (double *)shared; // shared memory для значений
    size_t *sIdx = (size_t *)&sdata[blockDim.x]; // shared memory для индексов

    size_t thrIdx = threadIdx.x;
    size_t i = blockDim.x * blockIdx.x + thrIdx;

    double maxValue = -1.0;
    size_t maxIdx = j;

    // второе условие - ограничение сетки для избежания неверных pivotCandidates[i]
    if (i < n && i >= j) {
        double value = fabs(a[i * n + j]);
        maxValue = value;
        maxIdx = i;
    }

    sdata[thrIdx] = maxValue;
    sIdx[thrIdx] = maxIdx;

    // до этой точки каждый поток заполнит ячейку в shared memory по своему индексу
    __syncthreads();

    // редукция по максимуму в пределах блока
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thrIdx < s && (thrIdx + s) < blockDim.x) {
            if (sdata[thrIdx] < sdata[thrIdx + s]) {
                sdata[thrIdx] = sdata[thrIdx + s];
                sIdx[thrIdx] = sIdx[thrIdx + s];
            }
        }
        __syncthreads();
    }

    // один поток блока записывает результат
    if (thrIdx == 0) {
        result[blockIdx.x] = sIdx[0]; // массов максимумов для каждого блока
    }
}

__global__ void normalizePivot(double *a, size_t n, size_t j) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t gridStride = gridDim.x * blockDim.x;

    for (size_t i = idx + j + 1; i < n; i += gridStride) {
        a[j * n + i] /= a[j * n + j];
    }
}

__global__ void eliminateBelow(double *a, size_t n, size_t j, const double eps) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t gridStride = gridDim.x * blockDim.x;

    size_t total = n * (n - j - 1); // число валидных пар (i, l)

    // представляем двойной цикл как один
    for (size_t k = idx; k < total; k += gridStride) {
        size_t i = k / (n - j - 1);
        size_t l = j + 1 + (k % (n - j - 1));

        if (j != i && fabs(a[i * n + j]) > eps) {
            a[i * n + l] -= a[j * n + l] * a[i * n + j];
        }
    }
}

double determinantGauss(double *matrix, size_t n) {
    if (!matrix) return 0;

    if (n == 1) return matrix[0];
    if (n == 2) return matrix[0] * matrix[3] - matrix[1] * matrix[2]; 

    double det = 1;
    const double eps = 1e-9;
    
    int deviceId;
    checkCuda( cudaGetDevice(&deviceId) );

    dim3 blockNum = getGridDim(n * n);
    dim3 threadNum = THREADS_PER_BLOCK;

    double *a = copyMatrix(matrix, n);
    
    size_t *pivotCandidates;
    checkCuda( cudaMallocManaged(&pivotCandidates, blockNum.x * sizeof(size_t)) );
    
    //checkCuda( cudaMemPrefetchAsync(a, n * n * sizeof(double), deviceId) );
    //checkCuda( cudaMemPrefetchAsync(pivotCandidates, blockNum.x * sizeof(size_t), deviceId) );

    for (size_t j = 0; j < n; j++) {
        // GPU kernel invocations
        // нужно обратить внимание на низкую ёмкость shared memory
        size_t sharedMemSize = threadNum.x * (sizeof(double) + sizeof(size_t)); // double + index
        findPivotIndex<<<getGridDim(n - j), threadNum, sharedMemSize>>>(a, n, j, pivotCandidates);
        checkCuda( cudaGetLastError() );

        checkCuda( cudaDeviceSynchronize() );

        // возвращаем на host для редукции
        //checkCuda( cudaMemPrefetchAsync(pivotCandidates, blockNum.x * sizeof(size_t), cudaCpuDeviceId) );
        //checkCuda( cudaMemPrefetchAsync(a, n * n * sizeof(double), cudaCpuDeviceId) );

        // редукция по блокам на CPU, можно переписать на GPU
        size_t k = j;
        for (size_t i = 0; i < blockNum.x; i++) {
            size_t cand = pivotCandidates[i];
            if (fabs(a[cand * n + j]) > fabs(a[k * n + j])) k = cand;
        }

        if (fabs(a[k * n + j]) < eps) {
            checkCuda( cudaFree(pivotCandidates) );
            checkCuda( cudaFree(a) );
            return 0;
        }

        //checkCuda( cudaMemPrefetchAsync(a, n * n * sizeof(double), deviceId) );

        if (j != k) {
            swap<<<getGridDim(n), threadNum>>>(a, n, j, k);
            checkCuda( cudaGetLastError() );

            det = -det;
            checkCuda( cudaDeviceSynchronize() );
        }
        det *= a[j * n + j];

        // нормализация ведущей строки
        normalizePivot<<<getGridDim(n), threadNum>>>(a, n, j);
        checkCuda( cudaGetLastError() );

        checkCuda( cudaDeviceSynchronize() );
    
        // обнуление элементов строк
        eliminateBelow<<<blockNum, threadNum>>>(a, n, j, eps);
        checkCuda( cudaGetLastError() );

        checkCuda( cudaDeviceSynchronize() );
    }

    checkCuda( cudaFree(pivotCandidates) );
    checkCuda( cudaFree(a) );
    return det;
}
// нормализацию и обнуление можно объединить

double minor(double *a, size_t n, size_t i, size_t j) {
    if (!a) return 0;

    size_t minorSize = n - 1;
    double *m = allocateMatrix(minorSize);
    if (!m) return 0;

    for (size_t row = 0, k = 0; row < n; row++) {
        if (row == i) continue;

        for (size_t col = 0, l = 0; col < n; col++) {
            if (col == j) continue;
    
            m[k * minorSize + l] = a[row * n + col];
            l++;
        }
        k++;
    }

    double mnr = determinantGauss(m, minorSize);
    checkCuda( cudaFree(m) );

    return mnr;
}

double *minorsMatrix(double *a, size_t n) {
    if (!a) return NULL;

    double *m = allocateMatrix(n);
    if (!m) return NULL;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            m[i * n + j] = minor(a, n, i, j);
        }
    }

    return m;
}

__global__ void algebraicAdditions(double *a, double *m, double *c, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t stride = gridDim.x + blockDim.x;

    for (size_t k = idx; k < n * n; k += stride) {
        size_t i = k / n;
        size_t j = k % n;
        
        c[i * n + j] = ((i + j) % 2 == 0 ? 1.0 : -1.0) * m[i * n + j];
    }
}

double *inversedFromGauss(double *a, size_t n) {
    const double eps = 1e-9;

    int deviceId;
    checkCuda( cudaGetDevice(&deviceId) );

    dim3 blockNum = getGridDim(n * n);
    dim3 threadNum = THREADS_PER_BLOCK;

    double det = determinantGauss(a, n); // A на CPU после выполнения
    if (fabs(det) < eps) return NULL;
    double revDet = 1 / det;

    double *m = minorsMatrix(a, n);
    double *c = allocateMatrix(n);
    //checkCuda( cudaMemPrefetchAsync(a, n * n * sizeof(double), deviceId) );
    //checkCuda( cudaMemPrefetchAsync(c, n * n * sizeof(double), deviceId) );

    algebraicAdditions<<<blockNum, threadNum>>>(a, m, c, n);
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );
    
    transpose<<<blockNum, threadNum>>>(c, n);
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );
    
    multiplyByNum<<<blockNum, threadNum>>>(c, n, revDet);
    checkCuda( cudaGetLastError() );
    checkCuda( cudaDeviceSynchronize() );

    //cudaMemPrefetchAsync(c, n * n * sizeof(double), cudaCpuDeviceId);
    
    checkCuda( cudaFree(m) );
    return c;
}

__global__ void initL(double *l, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t gridStride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < n ; i += gridStride) {
        l[i * n + i] = 1.0;
    }
}

__global__ void updateLU(double *a, double *l, double *u, size_t n, size_t i) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t gridStride = gridDim.x * blockDim.x;
    size_t tIdx = threadIdx.x;

    // динамическая shared memory
    extern __shared__ double shar[];
    double *lRow = shar;
    double *uCol = &shar[n]; // lRow[0:n], uCol[n:2n]

    if (tIdx< i) {
        lRow[tIdx] = l[i * n + tIdx];
        uCol[tIdx] = u[tIdx * n + i];
    }
    __syncthreads();

    for (size_t j = i + idx; j < n; j += gridStride) {
        // update U[i][j],
        double sumU = 0.0;
        for (size_t k = 0; k < i; k++) { 
            sumU += lRow[k] * u[k * n + j]; // l[i * n + k] * u[k * n + j];
        }
        u[i * n + j] = a[i * n + j] - sumU;

        // update L[j][i]
        double sumL = 0.0;
        for (size_t k = 0; k < i; k++) { 
            sumL += l[j * n + k] * uCol[k]; // l[j * n + k] * u[k * n + i];
        }
        l[j * n + i] = (a[j * n + i] - sumL) / u[i * n + i];

    }
}

void LUdecomposition(double *a, double *l, double *u, size_t n) {
    for (size_t i = 0; i < n; i++) {
        size_t blockNum = (n - 1 - i) / THREADS_PER_BLOCK + 1;
        size_t sharedSize = 2 * n * sizeof(double);

        updateLU<<<blockNum, THREADS_PER_BLOCK, sharedSize>>>(a, l, u, n, i);
    }
    cudaDeviceSynchronize();
}

double determinantLU(double *u, size_t n) {
    double det = 1.0;

    for (size_t i = 0; i < n; i++) {
        det *= u[i * n + i];
    }
    return det;
}

__global__ void inversedFromLU(double *l, double *u, double *inv, double *yAll, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    size_t gridStride = gridDim.x * blockDim.x;

    // цикл по столбцам
    for (size_t j = idx; j < n; j += gridStride) {
        double *y = &yAll[j * n]; // каждый поток работает с отдельной строкой

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
    }
}