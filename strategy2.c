#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h> // TODO: Check: is the include allowed?

void write_output(char fname[], double** arr, int n ){
    FILE *f = fopen(fname, "w");
    for( int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            fprintf(f, "%0.12f ", arr[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void crout(double const **A, double **L, double **U, int n) {
    // Let us write the code for only 2 threads first
    int i, j, k;
    double sum = 0;
    for (i = 0; i < n; i++) {
        U[i][i] = 1;
    }

    for (j = 0; j < n; j++) {
        // Compute L[j][j] first!
        sum = 0;
        for (k = 0; k < j; k++) {
            sum = sum + L[j][k] * U[k][j];
        }
        L[j][j] = A[j][j] - sum;

        if (L[j][j] == 0) {
            exit(0);
        }

#pragma omp parallel
        {
#pragma omp sections
            {
#pragma omp section
                {
                    // printf("39 %d\n",omp_get_thread_num() );
                    // This section just accesses the first j-1 rows of U
                    // Which have already been computed
                    // This writes the jth column of L!
                    int i1;
                    for (i1 = j+1; i1 < n; i1++) {
                        double sum1 = 0;
                        int k1;
                        for (k1 = 0; k1 < j; k1++) {
                            sum1 = sum1 + L[i1][k1] * U[k1][j];
                        }
                        L[i1][j] = A[i1][j] - sum1;
                    }
                }
#pragma omp section
                {
                    // printf("53 %d\n",omp_get_thread_num() );
                    // This just accesses the first j-1 cols of L
                    // Which have been already computed
                    // This writes the jth row of U!
                    int i2;
                    for (i2 = j; i2 < n; i2++) {
                        double sum2 = 0;
                        int k2;
                        for(k2 = 0; k2 < j; k2++) {
                            sum2 = sum2 + L[j][k2] * U[k2][i2];
                        }
                        U[j][i2] = (A[j][i2] - sum2) / L[j][j];
                    }
                }
            }
        }
    }
}

int main(int argc,char* argv[]){

    // gcc -o 0 -fopenmp strategy0.c

    int n,NUM_THREADS,loop2_threads,loop3_threads;
    n = atoi(argv[1]);
    char* input = argv[2];
    NUM_THREADS = atoi(argv[3]);

    double* A[n];
    double* L[n];
    double* U[n];

    // should we parallelize?
    for(int i=0;i<n;i++){
        A[i] = (double*)malloc(sizeof(double)*n);
        L[i] = (double*)malloc(sizeof(double)*n);
        U[i] = (double*)malloc(sizeof(double)*n);
    }

    FILE* f = fopen(input,"r");

    // now take input
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            fscanf(f,"%lf",&A[i][j]);
        }
    }

    // initialise L,U
    // should we parallelize?

    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            L[i][j]=0;
            U[i][j]=0;
        }
    }

    omp_set_num_threads(NUM_THREADS);
//    printf("Now Executing Crout!\n");
    crout((const double **) A, L, U, n);

    char str[100] = "output_L_2_";
    char intbuf[10];
    sprintf(intbuf,"%d",NUM_THREADS);
    strcat(str,intbuf);
    strcat(str,".txt");

    write_output(str,L,n);

    char str2[100] = "output_U_2_";
    strcat(str2,intbuf);
    strcat(str2,".txt");
    write_output(str2,U,n);
}