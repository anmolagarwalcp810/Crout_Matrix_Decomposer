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

void crout(double const **A, double **L, double **U, int n,int loop2_threads) {
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

        if(L[j][j]==0){
            exit(0);
        }
#pragma omp parallel num_threads(2)
    #pragma omp sections
                {
        #pragma omp section
                    {
                    // This section just accesses the first j-1 rows of U
                    // Which have already been computed
                    // This writes the jth column of L!
                    int i1,k1;
                    double sum1;
                    #pragma omp parallel for private(i1,k1,sum1) shared(j,A,L,U,n) schedule(static) num_threads(loop2_threads)
                    for (i1 = j+1; i1 < n; i1++) {
                        sum1 = 0;
                        for (k1 = 0; k1 < j; k1++) {
                            sum1 = sum1 + L[i1][k1] * U[k1][j];
                        }
                        L[i1][j] = A[i1][j] - sum1;
                    }
                }
        #pragma omp section
                    {
                    // This just accesses the first j-1 cols of L
                    // Which have been already computed
                    // This writes the jth row of U!
                    int i2,k2;
                    double sum2;
                    #pragma omp parallel for private(i2,k2,sum2) shared(j,A,L,U,n) schedule(static) num_threads(loop2_threads)
                    for (i2 = j; i2 < n; i2++) {
                        sum2 = 0;
                        for(k2 = 0; k2 < j; k2++) {
                            sum2 = sum2 + L[j][k2] * U[k2][i2];
                        }
                        U[j][i2] = (A[j][i2] - sum2) / L[j][j];
                    }
                    }
            }
    }
}

int main(int argc,char* argv[]){

    // gcc -o 0 -fopenmp strategy0.c

    int n,NUM_THREADS,loop2_threads;
    n = atoi(argv[1]);
    char* input = argv[2];
    NUM_THREADS = atoi(argv[3]);

    double* A[n];
    double* L[n];
    double* U[n];

    switch(NUM_THREADS){
        case(2):
            loop2_threads=1;
        case(4):
            loop2_threads=2;
        case(8):
            loop2_threads=4;
        case(16):
            loop2_threads=8;
    }

    // should we parallelize?
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
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

    #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            L[i][j]=0;
            U[i][j]=0;
        }
    }

    // omp_set_num_threads(NUM_THREADS);
//    printf("Now Executing Crout!\n");
    crout((const double **) A, L, U, n,loop2_threads);

    char str[100] = "output_L_3_";
    char intbuf[10];
    sprintf(intbuf,"%d",NUM_THREADS);
    strcat(str,intbuf);
    strcat(str,".txt");

    write_output(str,L,n);

    char str2[100] = "output_U_3_";
    strcat(str2,intbuf);
    strcat(str2,".txt");
    write_output(str2,U,n);
}