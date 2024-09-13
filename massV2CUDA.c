#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DATA_SIZE 10
#define QUERY_SIZE 4
#define ROWS 5



void generate_random_array(double array[ROWS][DATA_SIZE], int rows, int cols, int min, int max) {
    // Seed the random number generator
    srand(time(NULL));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            array[i][j] = min + (rand() % (max - min + 1));
        }
    }
}


double compute_mean(const double *y, int m) {
    double sum = 0.0;
    for (int i = 0; i < m; ++i) {
        sum += y[i];
    }
    return sum / m;
}

double compute_std_dev(const double *y, int m, double meany) {
    double sum_sq = 0.0;
    for (int i = 0; i < m; ++i) {
        double diff = y[i] - meany;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / m);
}

void compute_x_stats(const double *x, int m, double *mean_less_than_m, double *std_less_than_m) {
    double cumsum[m - 1];
    double square_sum[m - 1];
    double divider[m - 1];

    double sum = 0.0;
    for (int i = 0; i < m - 1; ++i) {
        sum += x[i];
        cumsum[i] = sum;
    }

    double sum_sq = 0.0;
    for (int i = 0; i < m - 1; ++i) {
        sum_sq += x[i] * x[i];
        square_sum[i] = sum_sq;
    }

    for (int i = 0; i < m - 1; ++i) {
        divider[i] = i + 1;
    }

    for (int i = 0; i < m - 1; ++i) {
        mean_less_than_m[i] = cumsum[i] / divider[i];
        double variance = (square_sum[i] - (cumsum[i] * cumsum[i]) / divider[i]) / divider[i];
        std_less_than_m[i] = sqrt(variance);
    }
}

void compute_sliding_window_stats(const double *x, int n, int m, double *mean_greater_than_m, double *std_greater_than_m) {
    for (int i = 0; i <= n - m; ++i) {
        double window[m];
        for (int j = 0; j < m; ++j) {
            window[j] = x[i + j];
        }
        double mean = compute_mean(window, m);
        double std_dev = compute_std_dev(window, m, mean);

        mean_greater_than_m[i] = mean;
        std_greater_than_m[i] = std_dev;
    }
}


void computeFFT(double *real_in, double *imag_in, double *real_out, double *imag_out, int N) {
    const double factor = 2 * M_PI / N;
    double angle, cos_val, sin_val;

    for (int k = 0; k < N; k++) {
        real_out[k] = 0.0;
        imag_out[k] = 0.0;

        for (int n = 0; n < N; n++) {
            angle = factor * k * n;
            cos_val = cos(angle);
            sin_val = sin(angle);

            real_out[k] += real_in[n] * cos_val + imag_in[n] * sin_val;
            imag_out[k] += -real_in[n] * sin_val + imag_in[n] * cos_val;
        }
    }
}

void complexMultiply(const double *real1, const double *imag1, const double *real2, const double *imag2,
                     double *result_real, double *result_imag, int N) {
    for (int i = 0; i < N; i++) {
        result_real[i] = real1[i] * real2[i] - imag1[i] * imag2[i];
        result_imag[i] = real1[i] * imag2[i] + imag1[i] * real2[i];
    }
}

void computeIFFT(double *real_in, double *imag_in, double *real_out, int N) {
    const double factor = 2 * M_PI / N;
    double angle, cos_val, sin_val;

    for (int k = 0; k < N; k++) {
        real_out[k] = 0.0;
        for (int n = 0; n < N; n++) {
            angle = factor * k * n;
            cos_val = cos(angle);
            sin_val = sin(angle);
            real_out[k] += real_in[n] * cos_val - imag_in[n] * sin_val;
        }
        real_out[k] /= N;
    }
}

void MASS_V2(double *x, double *y, int n, int m, double *dist) {
    double meany = compute_mean(y, m);
    double sigmay = compute_std_dev(y, m, meany);

    double *mean_less_than_m = (double*)malloc((m - 1) * sizeof(double));
    double *std_less_than_m = (double*)malloc((m - 1) * sizeof(double));
    if (mean_less_than_m == NULL || std_less_than_m == NULL) {
        printf("Memory allocation failed!\n");
        return;
    }

    compute_x_stats(x, m, mean_less_than_m, std_less_than_m);

    double *mean_greater_than_m = (double*)malloc((n - m + 1) * sizeof(double));
    double *std_greater_than_m = (double*)malloc((n - m + 1) * sizeof(double));
    if (mean_greater_than_m == NULL || std_greater_than_m == NULL) {
        printf("Memory allocation failed!\n");
        free(mean_less_than_m);
        free(std_less_than_m);
        return;
    }

    compute_sliding_window_stats(x, n, m, mean_greater_than_m, std_greater_than_m);

    double *meanx = (double*)malloc(n * sizeof(double));
    double *sigmax = (double*)malloc(n * sizeof(double));

    for (int i = 0; i < m - 1; ++i) {
        meanx[i] = mean_less_than_m[i];
        sigmax[i] = std_less_than_m[i];
    }

    for (int i = 0; i <= n - m; ++i) {
        meanx[m - 1 + i] = mean_greater_than_m[i];
        sigmax[m - 1 + i] = std_greater_than_m[i];
    }

    double *new_y = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < m; ++i) new_y[i] = y[m - 1 - i];
    for (int i = m; i < n; ++i) new_y[i] = 0.0;

    double *imag_in = (double*)calloc(n, sizeof(double));
    double *x_real_out = (double*)calloc(n, sizeof(double));
    double *x_imag_out = (double*)calloc(n, sizeof(double));
    double *y_real_out = (double*)calloc(n, sizeof(double));
    double *y_imag_out = (double*)calloc(n, sizeof(double));
    double *result_real = (double*)calloc(n, sizeof(double));
    double *result_imag = (double*)calloc(n, sizeof(double));
    double *ifft_real = (double*)calloc(n, sizeof(double));

    computeFFT(x, imag_in, x_real_out, x_imag_out, n); 
    computeFFT(new_y, imag_in, y_real_out, y_imag_out, n);
    complexMultiply(x_real_out, x_imag_out, y_real_out, y_imag_out, result_real, result_imag, n);
    computeIFFT(result_real, result_imag, ifft_real, n);

    for (int i = m - 1; i < n; ++i) {
        double mean_x = meanx[i];
        double sigma_x = sigmax[i];
        double z_value = ifft_real[i];
        
        double numerator = z_value - m * mean_x * meany;
        double denominator = sigma_x * sigmay;
        double value = 2 * (m - numerator / denominator);
        
        dist[i - (m - 1)] = fmax(value, 0.0);
    }

    for (int i = 0; i <= n - m; ++i) {
        dist[i] = sqrt(dist[i]);
    }

    free(mean_less_than_m);
    free(std_less_than_m);
    free(mean_greater_than_m);
    free(std_greater_than_m);
    free(meanx);
    free(sigmax);
    free(new_y);
    free(imag_in);
    free(x_real_out);
    free(x_imag_out);
    free(y_real_out);
    free(y_imag_out);
    free(result_real);
    free(result_imag);
    free(ifft_real);
}


int main() {
    double x[ROWS][DATA_SIZE] = {
        {1.0, 21.0, 43.0, 45.0, 15.0, 86.0, 75.0, 8.0, 9.0, 10.0},
        {145.6, 892.3, 234.8, 678.9, 102.5, 743.1, 399.7, 508.2, 926.4, 314.7},
        {823.4, 271.9, 459.6, 619.2, 354.8, 967.3, 785.1, 432.6, 573.8, 184.5},
        {2.4, 9.7, 72.5, 2.3, 4.1, 9.6, 7.8, 15.2, 46.9, 74.3},
        {48.5, 12.3, 59.7, 27.1, 93.8, 305.6, 487.4, 76.2, 95.1, 18.4}
    };


    // double x[ROWS][DATA_SIZE];
    // generate_random_array(x, ROWS, DATA_SIZE, 1, 1000);

    double y[] = {4.0, 12.0, 3.0, 19.0};

    int n = DATA_SIZE; // Length of x array
    int m = QUERY_SIZE; // Length of y array

    double (*distances)[DATA_SIZE - QUERY_SIZE + 1] = malloc(ROWS * (DATA_SIZE - QUERY_SIZE + 1) * sizeof(double));
    if (distances == NULL) {
        printf("Memory allocation failed!\n");
        return -1;
    }

    // Start timing
    clock_t start = clock();

    for (int row = 0; row < ROWS; ++row) {
        MASS_V2(x[row], y, n, m, distances[row]);
    }

    // End timing
    clock_t end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Full 2D array of distances:\n");
    for (int row = 0; row < ROWS; ++row) {
        for (int col = 0; col < DATA_SIZE - QUERY_SIZE + 1; ++col) {
            printf("%f \t", distances[row][col]);
        }
        printf("\n");
    }

    printf("Time elapsed for MASS_V2 function: %f seconds\n", elapsed_time);

    free(distances);

    return 0;
}
