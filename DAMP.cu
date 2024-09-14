#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_INSTANCES 5
#define MAX_QUERY_LENGTH 12
#define M_PI 3.14159265358979323846
#define TIME_SERIES_SIZE 50

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
    // Temporary arrays for FFT computations
    double *imag_in;
    double *x_real_out;
    double *x_imag_out;
    double *y_real_out;
    double *y_imag_out;
    double *result_real;
    double *result_imag;
    double *ifft_real;

    //for DAMP
    double *left_mp;
    double *bool_vec;
    double *distances;
    int *location;
    double *bsfScore;
};



#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)



void allocateDeviceMemory(struct nodeSOA *h_nodes, struct nodeSOA **d_nodes,int timeSeriesSize,int maxQuerySize) {
    // Allocate memory for the array of structs on the device
    CUDA_CHECK(cudaMalloc(d_nodes, NUM_INSTANCES * sizeof(struct nodeSOA)));

    for (int i = 0; i < NUM_INSTANCES; i++) {
        // Allocate memory for each member of the struct
        CUDA_CHECK(cudaMalloc(&h_nodes[i].x_less_than_m, maxQuerySize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].divider, (maxQuerySize - 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].cumsum_, maxQuerySize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].square_sum_less_than_m, maxQuerySize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].mean_less_than_m, (maxQuerySize - 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].std_less_than_m, (maxQuerySize - 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].windows, (timeSeriesSize - maxQuerySize + 1) * maxQuerySize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].mean_greater_than_m, (timeSeriesSize - maxQuerySize + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].std_greater_than_m, (timeSeriesSize - maxQuerySize + 1) * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].meanx, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].sigmax, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].y, maxQuerySize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].imag_in, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].x_real_out, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].x_imag_out, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].y_real_out, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].y_imag_out, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].result_real, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].result_imag, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].ifft_real, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].bool_vec, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].left_mp, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].distances, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].bsfScore,sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].location,sizeof(int)));
    }
}

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
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].imag_in, &h_nodes[i].imag_in, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].x_real_out, &h_nodes[i].x_real_out, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].x_imag_out, &h_nodes[i].x_imag_out, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].y_real_out, &h_nodes[i].y_real_out, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].y_imag_out, &h_nodes[i].y_imag_out, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].result_real, &h_nodes[i].result_real, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].result_imag, &h_nodes[i].result_imag, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].ifft_real, &h_nodes[i].ifft_real, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].left_mp, &h_nodes[i].left_mp, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].bool_vec, &h_nodes[i].bool_vec, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].distances, &h_nodes[i].distances, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].bsfScore, &h_nodes[i].bsfScore, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].location, &h_nodes[i].location, sizeof(int *), cudaMemcpyHostToDevice));
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

        // Free temporary arrays
        CUDA_CHECK(cudaFree(h_nodes[i].imag_in));
        CUDA_CHECK(cudaFree(h_nodes[i].x_real_out));
        CUDA_CHECK(cudaFree(h_nodes[i].x_imag_out));
        CUDA_CHECK(cudaFree(h_nodes[i].y_real_out));
        CUDA_CHECK(cudaFree(h_nodes[i].y_imag_out));
        CUDA_CHECK(cudaFree(h_nodes[i].result_real));
        CUDA_CHECK(cudaFree(h_nodes[i].result_imag));
        CUDA_CHECK(cudaFree(h_nodes[i].ifft_real));
        CUDA_CHECK(cudaFree(h_nodes[i].left_mp));
        CUDA_CHECK(cudaFree(h_nodes[i].bool_vec));
        CUDA_CHECK(cudaFree(h_nodes[i].distances));
        CUDA_CHECK(cudaFree(h_nodes[i].bsfScore));
        CUDA_CHECK(cudaFree(h_nodes[i].location));
    }
    // Free the device array of structs
    CUDA_CHECK(cudaFree(d_nodes));
}



int nextpow2(int x) {
    if (x <= 0) return 1; 
    int power = (int)ceil(log2((double)x));
    return (int)pow(2, power);
}

__global__ void DAMP(double *timeSeries,int start,int location_to_start,double best_so_far,int lookahead,int timeSeriesSize
,nodeSOA* nodes,double *distances_DAMP,double *location, double *bsf){
    int tid=threadIdx.x;
    nodeSOA node=nodes[tid];
    printf("%d ",tid+start);
    for(int i=0;i<timeSeriesSize;i++){
        node.left_mp[i]=0;
        node.bool_vec[i]=1;
    }
}




void printResults(struct nodeSOA *h_nodes) {
    for (int i = 0; i < NUM_INSTANCES; i++) {
        double *h_distances = (double *)malloc(TIME_SERIES_SIZE * sizeof(double));
        int *h_location = (int *)malloc(sizeof(int)); // Allocate memory for location
        double *h_bsfScore = (double *)malloc(sizeof(double)); // Allocate memory for bsfScore

        // Copy results from device to host
        cudaMemcpy(h_distances, h_nodes[i].distances, TIME_SERIES_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_location, h_nodes[i].location, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bsfScore, h_nodes[i].bsfScore, sizeof(double), cudaMemcpyDeviceToHost);

        // Print results
        printf("Instance %d:\n", i);
        for (int j = 0; j < TIME_SERIES_SIZE; j++) {
            printf("Index %d: distance = %f\n", j, h_distances[j]);
        }
        printf("Location: %d, bsfScore: %f\n", *h_location, *h_bsfScore);

        // Free host memory for this instance
        free(h_distances);
        free(h_location);
        free(h_bsfScore);
    }
}

int main(){

    int lookahead=25;
    int nextPower=nextpow2(lookahead);

    double h_array[TIME_SERIES_SIZE]={1.0, 21.0, 43.0, 45.0, 15.0, 86.0, 75.0, 8.0, 9.0, 10.0,145.6, 892.3, 
            234.8, 678.9, 102.5, 743.1, 399.7, 508.2, 926.4, 314.7,
            823.4, 271.9, 459.6, 619.2, 354.8, 967.3, 785.1, 432.6, 573.8, 184.5,
            2.4, 9.7, 72.5, 2.3, 4.1, 9.6, 7.8, 15.2, 46.9, 74.3,
            48.5, 12.3, 59.7, 27.1, 93.8, 305.6, 487.4, 76.2, 95.1, 18.4};
    double *d_array;
    cudaMalloc((void **)&d_array, TIME_SERIES_SIZE * sizeof(double));
    cudaMemcpy(d_array, h_array, TIME_SERIES_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    //For temp arrays for entire code
    struct nodeSOA *d_nodes;              
    struct nodeSOA h_nodes[NUM_INSTANCES]; 
    allocateDeviceMemory(h_nodes, &d_nodes,TIME_SERIES_SIZE,MAX_QUERY_LENGTH);
    copyHostToDevice(h_nodes, d_nodes);


    //For DAMP as we need all the final arrays MP of damp and bsf scores each and locations of each m sub query.
    size_t num_elements = NUM_INSTANCES *TIME_SERIES_SIZE;
    double *distance_DAMP = (double*)malloc(num_elements * sizeof(double));
    double *d_distances_DAMP;
    CUDA_CHECK(cudaMalloc((void**)&d_distances_DAMP, num_elements * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_distances_DAMP, distance_DAMP, num_elements * sizeof(double), cudaMemcpyHostToDevice));
    double *locations = (double*)malloc(NUM_INSTANCES*sizeof(double));
    double *d_locations;
    CUDA_CHECK(cudaMalloc((void**)&d_locations, NUM_INSTANCES * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_locations, locations, NUM_INSTANCES * sizeof(double), cudaMemcpyHostToDevice));
    double *bsfScores = (double*)malloc(NUM_INSTANCES*sizeof(double));
    double *d_bsfScores;
    CUDA_CHECK(cudaMalloc((void**)&d_bsfScores, NUM_INSTANCES * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_bsfScores, bsfScores, NUM_INSTANCES * sizeof(double), cudaMemcpyHostToDevice));
    


    DAMP<<<1, NUM_INSTANCES>>>(d_array,8,0,0,nextPower,TIME_SERIES_SIZE,d_nodes,d_distances_DAMP,d_locations,d_bsfScores);
    cudaDeviceSynchronize();


    printResults(h_nodes);

    freeDeviceMemory(h_nodes, d_nodes);
    return 0;
}