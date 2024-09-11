#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_INSTANCES 2
#define N 10
#define QUERY_LENGTH 4
#define M_PI 3.14159265358979323846

struct nodeSOA {
    double *x_less_than_m;
    double *divider;
    double *cumsum_;
    double *square_sum_less_than_m;
    double *mean_less_than_m;
    double *std_less_than_m;
    double *windows;
    double *mean_greater_than_m;
    double *std_greater_than_m;
    double *meanx;
    double *sigmax;
    double *y;
    double *X;
    double *Y;
    double *Z;
    double *z;
    // Temporary arrays for FFT computations
    double *imag_in;
    double *x_real_out;
    double *x_imag_out;
    double *y_real_out;
    double *y_imag_out;
    double *result_real;
    double *result_imag;
    double *ifft_real;
};


// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)


void allocateDeviceMemory(struct nodeSOA *h_nodes, struct nodeSOA **d_nodes) {
    // Allocate memory for the array of structs on the device
    CUDA_CHECK(cudaMalloc(d_nodes, NUM_INSTANCES * sizeof(struct nodeSOA)));

    for (int i = 0; i < NUM_INSTANCES; i++) {
        // Allocate memory for each member of the struct
        CUDA_CHECK(cudaMalloc(&h_nodes[i].x_less_than_m, QUERY_LENGTH * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].divider, (QUERY_LENGTH - 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].cumsum_, QUERY_LENGTH * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].square_sum_less_than_m, QUERY_LENGTH * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].mean_less_than_m, (QUERY_LENGTH - 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].std_less_than_m, (QUERY_LENGTH - 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].windows, (N - QUERY_LENGTH + 1) * QUERY_LENGTH * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].mean_greater_than_m, (N - QUERY_LENGTH + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].std_greater_than_m, (N - QUERY_LENGTH + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].meanx, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].sigmax, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].y, QUERY_LENGTH * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].X, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].Y, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].Z, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].z, N * sizeof(double)));

        // Allocate memory for temporary arrays
        CUDA_CHECK(cudaMalloc(&h_nodes[i].imag_in, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].x_real_out, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].x_imag_out, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].y_real_out, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].y_imag_out, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].result_real, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].result_imag, N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].ifft_real, N * sizeof(double)));
    }
}


// Function to copy pointers from the host struct to the device struct

void copyHostToDevice(struct nodeSOA *h_nodes, struct nodeSOA *d_nodes) {
    for (int i = 0; i < NUM_INSTANCES; i++) {
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].x_less_than_m, &h_nodes[i].x_less_than_m, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].divider, &h_nodes[i].divider, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].cumsum_, &h_nodes[i].cumsum_, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].square_sum_less_than_m, &h_nodes[i].square_sum_less_than_m, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].mean_less_than_m, &h_nodes[i].mean_less_than_m, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].std_less_than_m, &h_nodes[i].std_less_than_m, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].windows, &h_nodes[i].windows, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].mean_greater_than_m, &h_nodes[i].mean_greater_than_m, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].std_greater_than_m, &h_nodes[i].std_greater_than_m, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].meanx, &h_nodes[i].meanx, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].sigmax, &h_nodes[i].sigmax, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].y, &h_nodes[i].y, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].X, &h_nodes[i].X, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].Y, &h_nodes[i].Y, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].Z, &h_nodes[i].Z, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].z, &h_nodes[i].z, sizeof(double *), cudaMemcpyHostToDevice));

        // Copy pointers for additional temporary arrays
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].imag_in, &h_nodes[i].imag_in, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].x_real_out, &h_nodes[i].x_real_out, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].x_imag_out, &h_nodes[i].x_imag_out, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].y_real_out, &h_nodes[i].y_real_out, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].y_imag_out, &h_nodes[i].y_imag_out, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].result_real, &h_nodes[i].result_real, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].result_imag, &h_nodes[i].result_imag, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].ifft_real, &h_nodes[i].ifft_real, sizeof(double *), cudaMemcpyHostToDevice));
    }
}


void freeDeviceMemory(struct nodeSOA *h_nodes, struct nodeSOA *d_nodes) {
    for (int i = 0; i < NUM_INSTANCES; i++) {
        // Free memory for each member of the struct
        CUDA_CHECK(cudaFree(h_nodes[i].x_less_than_m));
        CUDA_CHECK(cudaFree(h_nodes[i].divider));
        CUDA_CHECK(cudaFree(h_nodes[i].cumsum_));
        CUDA_CHECK(cudaFree(h_nodes[i].square_sum_less_than_m));
        CUDA_CHECK(cudaFree(h_nodes[i].mean_less_than_m));
        CUDA_CHECK(cudaFree(h_nodes[i].std_less_than_m));
        CUDA_CHECK(cudaFree(h_nodes[i].windows));
        CUDA_CHECK(cudaFree(h_nodes[i].mean_greater_than_m));
        CUDA_CHECK(cudaFree(h_nodes[i].std_greater_than_m));
        CUDA_CHECK(cudaFree(h_nodes[i].meanx));
        CUDA_CHECK(cudaFree(h_nodes[i].sigmax));
        CUDA_CHECK(cudaFree(h_nodes[i].y));
        CUDA_CHECK(cudaFree(h_nodes[i].X));
        CUDA_CHECK(cudaFree(h_nodes[i].Y));
        CUDA_CHECK(cudaFree(h_nodes[i].Z));
        CUDA_CHECK(cudaFree(h_nodes[i].z));

        // Free temporary arrays
        CUDA_CHECK(cudaFree(h_nodes[i].imag_in));
        CUDA_CHECK(cudaFree(h_nodes[i].x_real_out));
        CUDA_CHECK(cudaFree(h_nodes[i].x_imag_out));
        CUDA_CHECK(cudaFree(h_nodes[i].y_real_out));
        CUDA_CHECK(cudaFree(h_nodes[i].y_imag_out));
        CUDA_CHECK(cudaFree(h_nodes[i].result_real));
        CUDA_CHECK(cudaFree(h_nodes[i].result_imag));
        CUDA_CHECK(cudaFree(h_nodes[i].ifft_real));
    }
    // Free the device array of structs
    CUDA_CHECK(cudaFree(d_nodes));
}


// Function to copy results from the device to the host
void copyResultsToHost(struct nodeSOA *h_nodes, struct nodeSOA *d_nodes, double *h_windows[]) {
    for (int i = 0; i < NUM_INSTANCES; i++) {
        h_windows[i] = (double *)malloc(N * sizeof(double));
        if (h_windows[i] == NULL) {
            fprintf(stderr, "Host memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        // Copy results from device to host
        CUDA_CHECK(cudaMemcpy(h_windows[i], h_nodes[i].windows, N * sizeof(double), cudaMemcpyDeviceToHost));
    }
}


__device__ double compute_mean(double *y, int m) {
    double sum = 0.0;
    for (int i = 0; i < m; ++i) {
        sum += y[i];
    }
    return sum / m;
}

__device__ double compute_std_dev(double *y, int m, double meany) {
    double sum_sq = 0.0;
    for (int i = 0; i < m; ++i) {
        double diff = y[i] - meany;
        sum_sq += diff * diff;
    }
    return sqrt(sum_sq / m);
}

__device__ void compute_x_stats(double *x, int m, nodeSOA node) {

    double sum = 0.0;
    for (int i = 0; i < m - 1; ++i) {
        sum += x[i];
        node.cumsum_[i]=sum;
    }

    double sum_sq = 0.0;
    for (int i = 0; i < m - 1; ++i) {
        sum_sq += x[i] * x[i];
        node.square_sum_less_than_m[i]=sum_sq;
    }

    for (int i = 0; i < m - 1; ++i) {
        node.divider[i]=i+1;
    }

    for (int i = 0; i < m - 1; ++i) {
        node.mean_less_than_m[i] = node.cumsum_[i] / node.divider[i];
        double variance = (node.square_sum_less_than_m[i] - (node.cumsum_[i] * node.cumsum_[i]) / node.divider[i]) / node.divider[i];
        node.std_less_than_m[i] = sqrt(variance);
    }
}


__device__ void compute_sliding_window_stats(double *x, int n, int m, nodeSOA node) {
    for (int i = 0; i <= n - m; ++i) {
        for (int j = 0; j < m; ++j) {
            node.windows[j]=x[i+j];
        }
        double mean = compute_mean(node.windows, m);
        double std_dev = compute_std_dev(node.windows, m, mean);

        node.mean_greater_than_m[i] = mean;
        node.std_greater_than_m[i] = std_dev;
    }
}

__device__ void computeFFT(double *real_in, double *imag_in, double *real_out, double *imag_out, int n) {
    const double factor = 2 * M_PI / n;
    double angle, cos_val, sin_val;
    for (int k = 0; k < n; k++) {
        real_out[k] = 0.0;
        imag_out[k] = 0.0;
        for (int i = 0; i < n; i++) {
            angle = factor * k * i;
            cos_val = cos(angle);
            sin_val = sin(angle);
            real_out[k] += real_in[i] * cos_val + imag_in[i] * sin_val;
            imag_out[k] += -real_in[i] * sin_val + imag_in[i] * cos_val;
        }
    }
}


__device__ void complexMultiply( double *real1,  double *imag1,  double *real2,  double *imag2,
                     double *result_real, double *result_imag, int n) {
    for (int i = 0; i < n; i++) {
        result_real[i] = real1[i] * real2[i] - imag1[i] * imag2[i];
        result_imag[i] = real1[i] * imag2[i] + imag1[i] * real2[i];
    }
}


__device__ void computeIFFT(double *real_in, double *imag_in, double *real_out) {
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


__global__ void massV2(double *timeSeries,double *query,nodeSOA* nodes,double *distances){
    int tid = threadIdx.x;
    double meany=compute_mean(query,QUERY_LENGTH);
    double sigmay=compute_std_dev(query,QUERY_LENGTH,meany);

    compute_x_stats(&timeSeries[tid*N],QUERY_LENGTH,nodes[tid]);
    compute_sliding_window_stats(&timeSeries[tid*N],N,QUERY_LENGTH,nodes[tid]);

    nodeSOA node=nodes[tid];
    for (int i = 0; i < QUERY_LENGTH - 1; ++i) {
        node.meanx[i] = node.mean_less_than_m[i];
        node.sigmax[i] = node.std_less_than_m[i];
    }

    for (int i = 0; i <= N - QUERY_LENGTH; ++i) {
        node.meanx[QUERY_LENGTH - 1 + i] = node.mean_greater_than_m[i];
        node.sigmax[QUERY_LENGTH - 1 + i] = node.std_greater_than_m[i];
    }

    for (int i = 0; i < QUERY_LENGTH; ++i) {
        node.y[i] = query[QUERY_LENGTH-i-1];
    }

    for (int i = QUERY_LENGTH; i < N; ++i) node.y[i] = 0.0;

    computeFFT(&timeSeries[tid],node.imag_in,node.x_real_out,node.x_imag_out,N);
    computeFFT(node.y,node.imag_in,node.y_real_out,node.y_imag_out,N);
    complexMultiply(node.x_real_out, node.x_imag_out, node.y_real_out,node. y_imag_out, node.result_real, node.result_imag,N);
    computeIFFT(node.result_real,node.result_imag,node.ifft_real);

    if(tid==0){
        printf("inversse fft result: \n");
        for(int i=0;i<N;i++){
            printf("%f ",node.ifft_real[i]);
        }
        printf("\n");
    }

    int row=tid*(N-QUERY_LENGTH+1);

    for (int i = QUERY_LENGTH - 1; i < N; ++i) {
        double mean_x = node.meanx[i];
        double sigma_x = node.sigmax[i];
        double z_value = node.ifft_real[i];
        
        double numerator = z_value - QUERY_LENGTH * mean_x * meany;
        double denominator = sigma_x * sigmay;
        double value = 2 * (QUERY_LENGTH - numerator / denominator);
        distances[row+(i - (QUERY_LENGTH - 1))] = fmax(value, 0.0);
    }

    for (int i = 0; i <= N - QUERY_LENGTH; ++i) {
        distances[row+i] = sqrt(distances[row+i]);
        // printf("%d %d %f",tid,i,distances[row+i]);
    }

}




int main() {
    double h_array[NUM_INSTANCES][N] = {
        {1.0, 21.0, 43.0, 45.0, 15.0, 86.0, 75.0, 8.0, 9.0, 10.0},
        {145.6, 892.3, 234.8, 678.9, 102.5, 743.1, 399.7, 508.2, 926.4, 314.7}
    };
    double *d_array;
    size_t array_size = NUM_INSTANCES * N * sizeof(double);
    cudaMalloc((void **)&d_array, array_size);
    cudaMemcpy(d_array, h_array, array_size, cudaMemcpyHostToDevice);


    double h_query[QUERY_LENGTH] = {4.0, 12.0, 3.0, 19.0};
    double *d_query;
    cudaMalloc((void **)&d_query, QUERY_LENGTH * sizeof(double));
    cudaMemcpy(d_query, h_query, QUERY_LENGTH * sizeof(double), cudaMemcpyHostToDevice);


    struct nodeSOA *d_nodes;              // Device array of structs
    struct nodeSOA h_nodes[NUM_INSTANCES]; // Host array of structs

    allocateDeviceMemory(h_nodes, &d_nodes);
    copyHostToDevice(h_nodes, d_nodes);

    size_t num_elements = NUM_INSTANCES * (N - QUERY_LENGTH + 1);
    double *distances = (double*)malloc(num_elements * sizeof(double));
    double *d_distances;
    CUDA_CHECK(cudaMalloc((void**)&d_distances, num_elements * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_distances, distances, num_elements * sizeof(double), cudaMemcpyHostToDevice));



    massV2<<<1, NUM_INSTANCES>>>(d_array,d_query,d_nodes,d_distances);
    cudaDeviceSynchronize();


    CUDA_CHECK(cudaMemcpy(distances, d_distances, num_elements * sizeof(double), cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUM_INSTANCES; i++) {
        printf("Row %d: ", i);
        for (int j = 0; j < (N - QUERY_LENGTH + 1); j++) {
            printf("%f ", distances[i * (N - QUERY_LENGTH + 1) + j]);
        }
        printf("\n");
    }


    freeDeviceMemory(h_nodes, d_nodes);
    CUDA_CHECK(cudaFree(d_distances));
    free(distances);

    return 0;
}
