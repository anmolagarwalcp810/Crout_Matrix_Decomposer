#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

void print_matrix(double **A,int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			printf("%0.12f ",A[i][j]);
		}
		printf("\n");
	}
}

void crout(double const **A, double **L, double **U, int n) {
	int i, j, k;
	double sum = 0;
	for (i = 0; i < n; i++) {
		U[i][i] = 1;
	} 
	for (j=0; j<n; j++) {
		for (i = j; i < n; i++) {
			sum = 0;
			for (k = 0; k < j; k++) {
				sum = sum + L[i][k] * U[k][j];
			} 
			L[i][j] = A[i][j] -	sum;
		} 
		for (i=j; i<n; i++) {
			sum = 0;
			for(k = 0; k < j; k++) {
				sum = sum + L[j][k] * U[k][i];
			} 
			// Why are we checking this multiple times? Check only once before entering the loop or rather, 
			// check on first update to avoid dependency
			if (L[j][j] == 0) {
				exit(0);
			} 
			U[j][i] =(A[j][i] -	sum) / L[j][j];
		}

	}
}

int main(int argc,char* argv[]){
	
	// gcc -o 0 -fopenmp strategy0.c

	int n,NUM_THREADS;
	// printf("%s\n",argv[0] );
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

	crout(A,L,U,n);

	// printf("A\n");
	// now check
	print_matrix(A,n);

	// printf("------------------------------\n");
	// printf("L\n");
	// output_(L/U)_<strategy(0/1/2/3/4)>_<number of threads/processes(2/4/8/16)>.txt
	char str[100] = "output_L_0_";
	char intbuf[10];
	sprintf(intbuf,"%d",NUM_THREADS);
	strcat(str,intbuf);
	strcat(str,".txt");
	
	write_output(str,L,n);

	// printf("------------------------------\n");
	// printf("U\n");
	char str2[100] = "output_U_0_";
	strcat(str2,intbuf);
	strcat(str2,".txt");
	write_output(str2,U,n);
}