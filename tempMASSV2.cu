#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 3
#define NUM_INSTANCES 100


//Blueprint of Global Memory to GPU

struct nodeSOA {
    int *square;
    double *cube;
};

// CUDA kernel to compute squares and cubes for all instances
__global__ void computeKernel(nodeSOA *nodes) {
    int tid = threadIdx.x;  // Each thread processes one instance

    if (tid < NUM_INSTANCES) {
        for (int i = 0; i < N; i++) {
            nodes[tid].square[i] = (tid + 1) * i * i;      // Compute square
            nodes[tid].cube[i] = (tid + 1) * i * i * i;    // Compute cube
        }
    }
}

int main() {
    nodeSOA *d_nodes;
    nodeSOA h_nodes[NUM_INSTANCES];

    // Allocate device memory for nodeSOA instances
    cudaMalloc(&d_nodes, NUM_INSTANCES * sizeof(nodeSOA));

    // Allocate memory for arrays in each instance on the device
    for (int i = 0; i < NUM_INSTANCES; i++) {
        cudaMalloc(&h_nodes[i].square, N * sizeof(int));
        cudaMalloc(&h_nodes[i].cube, N * sizeof(double));

        // Copy pointers from host instance to device instance
        cudaMemcpy(&d_nodes[i].square, &h_nodes[i].square, sizeof(int *), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_nodes[i].cube, &h_nodes[i].cube, sizeof(double *), cudaMemcpyHostToDevice);
    }

    // Launch kernel with one thread per instance
    computeKernel<<<1, NUM_INSTANCES>>>(d_nodes);
    cudaDeviceSynchronize();

    // Allocate host memory to retrieve results from the device
    for (int i = 0; i < NUM_INSTANCES; i++) {
        int *h_square = (int *)malloc(N * sizeof(int));
        double *h_cube = (double *)malloc(N * sizeof(double));

        // Copy results from device to host
        cudaMemcpy(h_square, h_nodes[i].square, N * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_cube, h_nodes[i].cube, N * sizeof(double), cudaMemcpyDeviceToHost);

        // Print results
        printf("Instance %d:\n", i);
        for (int j = 0; j < N; j++) {
            printf("Index %d: square = %d, cube = %f\n", j, h_square[j], h_cube[j]);
        }

        // Free host memory for this instance
        free(h_square);
        free(h_cube);
    }

    // Free allocated memory
    for (int i = 0; i < NUM_INSTANCES; i++) {
        cudaFree(h_nodes[i].square);
        cudaFree(h_nodes[i].cube);
    }
    cudaFree(d_nodes);

    return 0;
}
