#include <stdio.h>
#include <stdlib.h>

#define ROWS 5
#define QUERY_LENGTH 5
#define DATA_SIZE 10

// Define a struct to hold all the temporary arrays with fixed sizes
typedef struct {
    double *x_less_than_m;           // Size: m - 1
    double *divider;                 // Size: m - 1
    double *cumsum_;                 // Size: m - 1
    double *square_sum_less_than_m; // Size: m - 1
    double *mean_less_than_m;        // Size: m - 1
    double *std_less_than_m;         // Size: m - 1
    double *windows;                 // Size: (n - m + 1) * m
    double *mean_greater_than_m;    // Size: n - m + 1
    double *std_greater_than_m;     // Size: n - m + 1
    double *meanx;                   // Size: n
    double *sigmax;                  // Size: n
    double *y;                       // Size: n
    double *X;                       // Size: n
    double *Y;                       // Size: n
    double *Z;                       // Size: n
    double *z;                       // Size: n
} TempArrays;

// Function to allocate memory for the struct
TempArrays* allocate_temp_arrays(int m, int n) {
    TempArrays *temp = (TempArrays*)malloc(sizeof(TempArrays));
    if (!temp) {
        perror("Failed to allocate memory for TempArrays");
        exit(EXIT_FAILURE);
    }

    temp->x_less_than_m = (double*)malloc((m - 1) * sizeof(double));
    temp->divider = (double*)malloc((m - 1) * sizeof(double));
    temp->cumsum_ = (double*)malloc((m - 1) * sizeof(double));
    temp->square_sum_less_than_m = (double*)malloc((m - 1) * sizeof(double));
    temp->mean_less_than_m = (double*)malloc((m - 1) * sizeof(double));
    temp->std_less_than_m = (double*)malloc((m - 1) * sizeof(double));
    temp->windows = (double*)malloc((n - m + 1) * m * sizeof(double));
    temp->mean_greater_than_m = (double*)malloc((n - m + 1) * sizeof(double));
    temp->std_greater_than_m = (double*)malloc((n - m + 1) * sizeof(double));
    temp->meanx = (double*)malloc(n * sizeof(double));
    temp->sigmax = (double*)malloc(n * sizeof(double));
    temp->y = (double*)malloc(n * sizeof(double));
    temp->X = (double*)malloc(n * sizeof(double));
    temp->Y = (double*)malloc(n * sizeof(double));
    temp->Z = (double*)malloc(n * sizeof(double));
    temp->z = (double*)malloc(n * sizeof(double));

    return temp;
}

// Function to free the allocated memory
void free_temp_arrays(TempArrays *temp) {
    free(temp->x_less_than_m);
    free(temp->divider);
    free(temp->cumsum_);
    free(temp->square_sum_less_than_m);
    free(temp->mean_less_than_m);
    free(temp->std_less_than_m);
    free(temp->windows);
    free(temp->mean_greater_than_m);
    free(temp->std_greater_than_m);
    free(temp->meanx);
    free(temp->sigmax);
    free(temp->y);
    free(temp->X);
    free(temp->Y);
    free(temp->Z);
    free(temp->z);
    free(temp);
}

void print_array(double *array, int length, const char *name) {
    printf("%s:\n", name);
    for (int i = 0; i < length; i++) {
        printf("%f ", array[i]);
    }
    printf("\n");
}


void print_temp_arrays(TempArrays *temp, int m, int n) {
    // Print each array with a descriptive name
    print_array(temp->x_less_than_m, m - 1, "x_less_than_m");
    print_array(temp->divider, m - 1, "divider");
    print_array(temp->cumsum_, m - 1, "cumsum_");
    print_array(temp->square_sum_less_than_m, m - 1, "square_sum_less_than_m");
    print_array(temp->mean_less_than_m, m - 1, "mean_less_than_m");
    print_array(temp->std_less_than_m, m - 1, "std_less_than_m");
    print_array(temp->mean_greater_than_m, n - m + 1, "mean_greater_than_m");
    print_array(temp->std_greater_than_m, n - m + 1, "std_greater_than_m");
    print_array(temp->meanx, n, "meanx");
    print_array(temp->sigmax, n, "sigmax");
    print_array(temp->y, n, "y");
    print_array(temp->X, n, "X");
    print_array(temp->Y, n, "Y");
    print_array(temp->Z, n, "Z");
    print_array(temp->z, n, "z");

    // Special handling for the windows array (2D)
    printf("windows:\n");
    for (int i = 0; i < n - m + 1; i++) {
        for (int j = 0; j < m; j++) {
            printf("%f ", temp->windows[i * m + j]);
        }
        printf("\n");
    }
}


int main() {
    int m = 100;  // Example value for m
    int n = 1000; // Example value for n

    TempArrays* temp[ROWS];
    
    for (int i = 0; i < ROWS; i++){
       temp[i]=allocate_temp_arrays(QUERY_LENGTH,DATA_SIZE);
    }
    

    for (int i = 0; i < 1; i++){
        print_temp_arrays(temp[i], QUERY_LENGTH, DATA_SIZE);
    }

    for (int i = 0; i < ROWS; i++){
        free_temp_arrays(temp[i]);
    }


    return 0;
}
