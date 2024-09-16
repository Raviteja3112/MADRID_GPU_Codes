#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_INSTANCES 3
#define MAX_QUERY_LENGTH 10
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
    double *query;
    double *dp_mass_instance;

    //for DAMP
    double *left_mp;
    double *bool_vec;
    double *distances_DAMP;
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
        CUDA_CHECK(cudaMalloc(&h_nodes[i].query, maxQuerySize * sizeof(double)));
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
        CUDA_CHECK(cudaMalloc(&h_nodes[i].distances_DAMP, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].dp_mass_instance, timeSeriesSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].bsfScore,sizeof(double)));
        CUDA_CHECK(cudaMalloc(&h_nodes[i].location,sizeof(int)));
    }
}

void copyHostToDevice(struct nodeSOA *h_nodes, struct nodeSOA *d_nodes) {
    for (int i = 0; i < NUM_INSTANCES; i++) {
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].x_less_than_m, &h_nodes[i].x_less_than_m, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].query, &h_nodes[i].query, sizeof(double *), cudaMemcpyHostToDevice));
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
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].distances_DAMP, &h_nodes[i].distances_DAMP, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].dp_mass_instance, &h_nodes[i].dp_mass_instance, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].bsfScore, &h_nodes[i].bsfScore, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(&d_nodes[i].location, &h_nodes[i].location, sizeof(int *), cudaMemcpyHostToDevice));
    }
}


void freeDeviceMemory(struct nodeSOA *h_nodes, struct nodeSOA *d_nodes) {
    for (int i = 0; i < NUM_INSTANCES; i++) {
        // Free memory for each member of the struct
        CUDA_CHECK(cudaFree(h_nodes[i].x_less_than_m));
        CUDA_CHECK(cudaFree(h_nodes[i].divider));
        CUDA_CHECK(cudaFree(h_nodes[i].query));
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
        CUDA_CHECK(cudaFree(h_nodes[i].distances_DAMP));
        CUDA_CHECK(cudaFree(h_nodes[i].dp_mass_instance));
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


__device__ void compute_x_stats(double *x, int m, nodeSOA node,int timeSeriesStart) {

    double sum = 0.0;
    for (int i = 0; i < m - 1; ++i) {
        sum += x[timeSeriesStart+i];
        node.cumsum_[i]=sum;
    }

    double sum_sq = 0.0;
    for (int i = 0; i < m - 1; ++i) {
        sum_sq += x[timeSeriesStart+i] * x[timeSeriesStart+i];
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


__device__ void compute_sliding_window_stats(double *x, int n, int m, nodeSOA node,int timeSeriesStart) {
    for (int i = 0; i <= n - m; ++i) {
        for (int j = 0; j < m; ++j) {
            node.windows[j]=x[timeSeriesStart+i+j];
        }
        double mean = compute_mean(node.windows, m);
        double std_dev = compute_std_dev(node.windows, m, mean);

        node.mean_greater_than_m[i] = mean;
        node.std_greater_than_m[i] = std_dev;
    }
}

__device__ void computeFFT(double *real_in, double *imag_in, double *real_out, double *imag_out, int n,int start) {
    const double factor = 2 * M_PI / n;
    double angle, cos_val, sin_val;
    for (int k = 0; k < n; k++) {
        real_out[k] = 0.0;
        imag_out[k] = 0.0;
        for (int i = 0; i < n; i++) {
            angle = factor * k * i;
            cos_val = cos(angle);
            sin_val = sin(angle);
            real_out[k] += real_in[start+i] * cos_val + imag_in[i] * sin_val;
            imag_out[k] += -real_in[start+i] * sin_val + imag_in[i] * cos_val;
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


__device__ void computeIFFT(double *real_in, double *imag_in, double *real_out,int timeSeriesSize) {
    const double factor = 2 * M_PI / timeSeriesSize;
    double angle, cos_val, sin_val;

    for (int k = 0; k < timeSeriesSize; k++) {
        real_out[k] = 0.0;
        for (int n = 0; n < timeSeriesSize; n++) {
            angle = factor * k * n;
            cos_val = cos(angle);
            sin_val = sin(angle);
            real_out[k] += real_in[n] * cos_val - imag_in[n] * sin_val;
        }
        real_out[k] /= timeSeriesSize;
    }
}


__device__ void MASS_V2(double *timeSeries,int start,int end,nodeSOA node,int QUERY_LENGTH){
    double meany=compute_mean(node.query,QUERY_LENGTH);
    double sigmay=compute_std_dev(node.query,QUERY_LENGTH,meany);

    int timeSeriesSize=end-start+1;


    compute_x_stats(timeSeries,QUERY_LENGTH,node,start);
    compute_sliding_window_stats(timeSeries,timeSeriesSize,QUERY_LENGTH,node,start);

    for (int i = 0; i < QUERY_LENGTH - 1; ++i) {
        node.meanx[i] = node.mean_less_than_m[i];
        node.sigmax[i] = node.std_less_than_m[i];
    }

    for (int i = 0; i <= timeSeriesSize - QUERY_LENGTH; ++i) {
        node.meanx[QUERY_LENGTH - 1 + i] = node.mean_greater_than_m[i];
        node.sigmax[QUERY_LENGTH - 1 + i] = node.std_greater_than_m[i];
    }

    for (int i = 0; i < QUERY_LENGTH; ++i) {
        node.y[i] = node.query[QUERY_LENGTH-i-1];
    }

    for (int i = QUERY_LENGTH; i < timeSeriesSize; ++i) node.y[i] = 0.0;

    
    computeFFT(timeSeries,node.imag_in,node.x_real_out,node.x_imag_out,timeSeriesSize,start);
    computeFFT(node.y,node.imag_in,node.y_real_out,node.y_imag_out,timeSeriesSize,0);
    complexMultiply(node.x_real_out, node.x_imag_out, node.y_real_out,node. y_imag_out, node.result_real, node.result_imag,timeSeriesSize);
    computeIFFT(node.result_real,node.result_imag,node.ifft_real,timeSeriesSize);
  
    for (int i = QUERY_LENGTH - 1; i < timeSeriesSize; ++i) {
        double mean_x = node.meanx[i];
        double sigma_x = node.sigmax[i];
        double z_value = node.ifft_real[i];
        
        double numerator = z_value - QUERY_LENGTH * mean_x * meany;
        double denominator = sigma_x * sigmay;
        double value = 2 * (QUERY_LENGTH - numerator / denominator);
        node.dp_mass_instance[start+i - (QUERY_LENGTH - 1)] = fmax(value, 0.0);
    }

    for (int i = 0; i <= timeSeriesSize - QUERY_LENGTH; ++i) {
        node.dp_mass_instance[start+i] = sqrt(node.dp_mass_instance[start+i]);
    }

}


__global__ void DAMP(double *timeSeries,int subSequent_m_start,int location_to_start,int lookahead,int timeSeriesSize,nodeSOA* nodes){
    int tid=threadIdx.x;
    int subsequence_length=tid+subSequent_m_start;
    nodeSOA node=nodes[tid];
    printf("%d ",tid+subSequent_m_start);
    for(int i=0;i<timeSeriesSize;i++){
        node.left_mp[i]=0;
        node.bool_vec[i]=1;
    }

    

    for(int i=location_to_start-1;i<location_to_start+16*subsequence_length;i++){
        if(node.bool_vec[i]==0){
            node.left_mp[i]=node.left_mp[i-1]-1e-05;
            continue;
        }
        if(i+subsequence_length-1>timeSeriesSize)break;
        int index=0;
        for(int j=i;j<i+subsequence_length;j++){
            node.query[index++]=timeSeries[j];
        }
        MASS_V2(timeSeries,0,i,node,subsequence_length);
        double tempMin=node.dp_mass_instance[0];
        for(int j=1;j<i-subsequence_length+1;j++){
            if(tempMin>node.dp_mass_instance[j]){
                tempMin=node.dp_mass_instance[j];
            }
        }
        node.left_mp[i]=tempMin;
        double tempMax=node.left_mp[0];
        for(int j=1;j<i;j++){
            if(tempMax<node.left_mp[j]){
                tempMax=node.left_mp[j];
            }
        }
        *node.bsfScore=tempMax;

        if(lookahead!=0){
            int start_of_mass=i+subsequence_length-1<timeSeriesSize?i+subsequence_length-1:timeSeriesSize;
            int end_of_mass=start_of_mass+lookahead-1<timeSeriesSize?start_of_mass+lookahead-1:timeSeriesSize;
            if(end_of_mass-start_of_mass+1>subsequence_length){
                //double *timeSeries,int start,int end,nodeSOA node,int QUERY_LENGTH
                MASS_V2(timeSeries,start_of_mass,end_of_mass,node,subsequence_length);

                for(int dp_instance=start_of_mass;dp_instance<end_of_mass;dp_instance++){
                    if(node.dp_mass_instance[dp_instance]<*node.bsfScore){
                        node.bool_vec[dp_instance]=0;
                    }
                }
            }
        }

        
    }
}




void printResults(struct nodeSOA *h_nodes) {
    for (int i = 0; i < NUM_INSTANCES; i++) {
        // double *h_distances = (double *)malloc(TIME_SERIES_SIZE * sizeof(double));
        int *h_location = (int *)malloc(sizeof(int)); // Allocate memory for location
        double *h_bsfScore = (double *)malloc(sizeof(double)); // Allocate memory for bsfScore

        // Copy results from device to host
        // cudaMemcpy(h_distances, h_nodes[i].distances_DAMP, TIME_SERIES_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_location, h_nodes[i].location, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bsfScore, h_nodes[i].bsfScore, sizeof(double), cudaMemcpyDeviceToHost);

        // Print results
        printf("Instance %d:\n", i);
        // for (int j = 0; j < TIME_SERIES_SIZE; j++) {
        //     printf("Index %d: distance = %f\n", j, h_distances[j]);
        // }
        printf("Location: %d, bsfScore: %f\n", *h_location, *h_bsfScore);

        // Free host memory for this instance
        // free(h_distances);
        free(h_location);
        free(h_bsfScore);
    }
}

int main(){
    int lookahead=25;
    int nextPower=nextpow2(lookahead);
    int subseq=3;
    int location_to_process=10;

    double h_array[TIME_SERIES_SIZE]={1.0, 21.0, 43.0, 45.0, 15.0, 86.0, 75.0, 8.0, 9.0, 10.0, 15.6, 82.3, 
               2.4, 9.7, 72.5, 2.3, 4.1, 9.6, 7.8, 15.2, 46.9, 74.3,
               48.5, 12.3, 59.7, 27.1, 93.8, 305.6, 487.4, 76.2, 95.1, 18.4,
               234.8, 678.9, 102.5, 743.1, 399.7, 508.2, 926.4, 314.7,
               823.4, 271.9, 459.6, 619.2, 354.8, 967.3, 785.1, 432.6, 573.8, 184.5};
    double *d_array;
    cudaMalloc((void **)&d_array, TIME_SERIES_SIZE * sizeof(double));
    cudaMemcpy(d_array, h_array, TIME_SERIES_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    //For temp arrays for entire code
    struct nodeSOA *d_nodes;              
    struct nodeSOA h_nodes[NUM_INSTANCES]; 
    allocateDeviceMemory(h_nodes, &d_nodes,TIME_SERIES_SIZE,MAX_QUERY_LENGTH);
    copyHostToDevice(h_nodes, d_nodes);


    //For DAMP as we need all the final arrays MP of damp and bsf scores each and locations of each m sub query


    DAMP<<<1, NUM_INSTANCES>>>(d_array,subseq,location_to_process,nextPower,TIME_SERIES_SIZE,d_nodes);
    cudaDeviceSynchronize();


    printResults(h_nodes);

    freeDeviceMemory(h_nodes, d_nodes);
    return 0;
}