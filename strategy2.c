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

#pragma omp parallel
        {
#pragma omp sections
            {
#pragma omp section
                {
                    // This section just accesses the first j-1 rows of U
                    // Which have already been computed
                    // This writes the jth column of L!
                    for (i = j+1; i < n; i++) {
                        sum = 0;
                        for (k = 0; k < j; k++) {
                            sum = sum + L[i][k] * U[k][j];
                        }
                        L[i][j] = A[i][j] - sum;
                    }
                }
#pragma omp section
                {
                    // This just accesses the first j-1 cols of L
                    // Which have been already computed
                    // This writes the jth row of U!
                    for (i = j; i < n; i++) {
                        sum = 0;
                        for(k = 0; k < j; k++) {
                            sum = sum + L[j][k] * U[k][i];
                        }
                        /// Shouldn't we check this before the for-loop itself?
                        if (L[j][j] == 0) {
                            exit(0);
                        }
                        U[j][i] = (A[j][i] - sum) / L[j][j];
                    }
                }
            }
        }
    }
}

//int main(){
//    int n = 3;
//
//    double **A = (double **)malloc(n*sizeof(double*));
//    for(int i=0;i<n;i++){
//        A[i] = (double *)malloc(n*sizeof(double));
//    }
//
//    double **U = (double **)malloc(n*sizeof(double*));
//    for(int i=0;i<n;i++){
//        U[i] = (double *)malloc(n*sizeof(double));
//    }
//
//    double **L = (double **)malloc(n*sizeof(double*));
//    for(int i=0;i<n;i++){
//        L[i] = (double *)malloc(n*sizeof(double));
//    }
//
//    for(int i=0;i<n;i++){
//        for(int j=0;j<n;j++){
//            A[i][j] = i+j;
//            U[i][j] = 0;
//            L[i][j] = 0;
//        }
//    }
//
//
//    crout((const double **) A, L, U, n);
//}

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

#pragma omp parallel for schedule(static) num_threads(loop2_threads)
    for(int i=0;i<n;i++){
#pragma omp parallel for schedule(static) num_threads(loop3_threads)
        for(int j=0;j<n;j++){
            L[i][j]=0;
            U[i][j]=0;
        }
    }

    omp_set_num_threads(NUM_THREADS);
//    printf("Now Executing Crout!\n");
    crout((const double **) A, L, U, n);

    char str[100] = "output_L_1_";
    char intbuf[10];
    sprintf(intbuf,"%d",NUM_THREADS);
    strcat(str,intbuf);
    strcat(str,".txt");

    write_output(str,L,n);

    char str2[100] = "output_U_1_";
    strcat(str2,intbuf);
    strcat(str2,".txt");
    write_output(str2,U,n);
}