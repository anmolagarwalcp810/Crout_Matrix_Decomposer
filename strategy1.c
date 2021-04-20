#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

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

void summation_matrix_L(double** L,double** U,double* sum,int i,int j){
	double local_sum = 0;
	int thread_number = omp_get_thread_num();
	int num_threads = omp_get_num_threads();
	int range = j/num_threads;
	int left = thread_number*range;
	if(thread_number==num_threads-1){	// last thread, assuming j not always divisble by num_threads
		range = j-(num_threads-1)*range;
	}
	if(range==0){
		return;
	}
	int right = left+range;

	for(int k=left;k<right;k++){
		local_sum = local_sum + L[i][k] * U[k][j];
	}

	#pragma omp critical
	{
		// only 1 access to global variable sum
		*sum = *sum + local_sum;
	}
}

void summation_matrix_U(double** L,double** U,double *sum,int i,int j){
	double local_sum = 0;
	int thread_number = omp_get_thread_num();
	int num_threads = omp_get_num_threads();
	int range = j/num_threads;
	int left = thread_number*range;
	if(thread_number==num_threads-1){	// last thread, assuming j not always divisble by num_threads
		range = j-(num_threads-1)*range;
	}
	int right = left+range;

	for(int k=left;k<right;k++){
		local_sum = local_sum + L[j][k] * U[k][i];
	}

	#pragma omp critical
	{
		// only 1 access to global variable sum
		*sum = *sum + local_sum;
	}
}

void print_matrix(double **A,int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			printf("%0.12f ",A[i][j]);
		}
		printf("\n");
	}
}

void crout(double const **A, double **L, double **U, int n,int loop2_threads) {
	int i, j, k;
	double sum = 0;
	#pragma omp parallel for private(i) shared(U,n) schedule(static)
		for (i = 0; i < n; i++) {
			U[i][i] = 1;
		} 

	// implicit barrier

	/*
		Three nested loops and 2 threads in each would lead to 2^2 = 4 threads in total
		And, we are given atleast 8 cores.
	*/

	// later need to also decide the distribution among the three loops after taking thread count as the input

	// #pragma omp parallel for private(i,j,k,sum) shared(A,L,U,n) schedule(dynamic) num_threads(2)
		for (j=0; j<n; j++) {
			// if(j%100==0)printf("j:%d\n",j );
			#pragma omp parallel for private(i,k,sum) shared(j,A,L,U,n) schedule(static) num_threads(loop2_threads)
				for (i = j; i < n; i++) {
					// printf("2: i,%d; j,%d; thread,%d\n", i,j,omp_get_thread_num());
					sum = 0;
					for (k = 0; k < j; k++) {
						sum = sum + L[i][k] * U[k][j];
					} 
					L[i][j] = A[i][j] -	sum;
				}
			if(L[j][j]==0){
				exit(0);
			}
			#pragma omp parallel for private(i,k,sum) shared(j,A,L,U,n) schedule(static) num_threads(loop2_threads)
				for (i=j; i<n; i++) {
					// printf("4: i,%d; j,%d; thread,%d\n", i,j,omp_get_thread_num());
					sum = 0;
					for(k = 0; k < j; k++) {
						sum = sum + L[j][k] * U[k][i];
					} 
					U[j][i] =(A[j][i] -	sum) / L[j][j];
				}
		}
}

int main(int argc,char* argv[]){
	
	// gcc -o 0 -fopenmp strategy0.c

	/*
		Strategy 1 almost complete, now simply need to take input and test for both strategy 1 and 0.
		On terminal "export OMP_NESTED=TRUE", to be included in run.sh

	*/

	int n,NUM_THREADS;
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

	#pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			L[i][j]=0;
			U[i][j]=0;
		}
	}

	double start,end;
	// start = omp_get_wtime();
	// printf("start: %lf\n",start);

	crout(A,L,U,n,NUM_THREADS);

	// end = omp_get_wtime();

	// printf("TIME: %lf\n",end-start );
	// printf("A\n");
	// now check
	// print_matrix(A,n);

	// printf("------------------------------\n");
	// printf("L\n");
	// output_(L/U)_<strategy(0/1/2/3/4)>_<number of threads/processes(2/4/8/16)>.txt
	// print_matrix(L,n);
	char str[100] = "output_L_1_alt_";
	char intbuf[10];
	sprintf(intbuf,"%d",NUM_THREADS);
	strcat(str,intbuf);
	strcat(str,".txt");
	
	write_output(str,L,n);

	// printf("------------------------------\n");
	// printf("U\n");
	// print_matrix(U,n);
	char str2[100] = "output_U_1_alt_";
	strcat(str2,intbuf);
	strcat(str2,".txt");
	write_output(str2,U,n);
}