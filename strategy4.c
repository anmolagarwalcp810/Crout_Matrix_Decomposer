#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

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

void write_output_Transpose(char fname[], double** arr, int n ){
	FILE *f = fopen(fname, "w");
	for( int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			fprintf(f, "%0.12f ", arr[j][i]);
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

void print_matrix2(double A[4][4],int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			printf("%0.12f ",A[i][j]);
		}
		printf("\n");
	}
}

void print_matrix_transpose(double **A,int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			printf("%0.12f ",A[j][i]);
		}
		printf("\n");
	}
}

void crout(double const **A, double **L, double **U, int n,int my_rank,int comm_sz) {
	int i, j, k;
	double sum = 0;

	for (j=0; j<n; j++) {
		// First do with scatter, then try with scatterv() later.
		for (i = j; i < n; i++) {
			sum = 0;
			for (k = 0; k < j; k++) {
				sum = sum + L[i][k] * U[k][j];
			} 
			L[i][j] = A[i][j] -	sum;
			if(i==j){
				if (L[j][j] == 0) {
					exit(0);
				} 
			}
		} 
		for (i=j; i<n; i++) {
			sum = 0;
			for(k = 0; k < j; k++) {
				sum = sum + L[j][k] * U[k][i];
			} 
			U[j][i] =(A[j][i] -	sum) / L[j][j];
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}
}

int main(int argc,char* argv[]){
	int n;
	int comm_sz;	// number of process
	int my_rank;	// my process rank


	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	

	n = atoi(argv[1]);
	
	double A[n][n];
	double L[n][n];
	double U[n][n];

	if(my_rank==0){
		
		char* input = argv[2];
		
		FILE* f = fopen(input,"r");
		for(int i=0;i<n*n;i++){
			for(int j=0;j<n;j++){
				fscanf(f,"%lf",&A[i][j]);
			}
		}

		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				L[i][j]=0;
				if(i==j){
					U[i][j]=1;
				}
				else{
					U[i][j]=0;
				}
			}
		}

		// take transpose of U
	}	
	MPI_Bcast(&(A[0][0]),n*n,MPI_DOUBLE,0,MPI_COMM_WORLD);

	// CROUT MATRIX DECOMPOSITION STARTS HERE

	int i, j, k;
	double sum = 0;

	int chunk = n/comm_sz;
	double sub_L[chunk][n];
	double sub_U[chunk][n];
	int left_i,right_i;
	MPI_Barrier(MPI_COMM_WORLD);

	for (j=0; j<n; j++) {
		// First do with scatter, then try with scatterv() later.
		// printf("j: %d\n",j);
		MPI_Scatter(&(L[0][0]),chunk*n,MPI_DOUBLE,sub_L,chunk*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(U[j],n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		/* Allocating iteration ranges */
		if(my_rank*chunk>=j){
			left_i=0;
			right_i=chunk;
		}
		else{
			if(j-my_rank*chunk<chunk){
				left_i=j-my_rank*chunk;
				right_i=chunk;
			}
			else{
				left_i=0;
				right_i=-1;
			}
		}

		// L's Loop
		for (i = left_i; i < right_i; i++) {
			sum = 0;
			for (k = 0; k < j; k++) {
				sum = sum + sub_L[i][k] * U[j][k];
			}
			// printf("my_rank %d, i %d,j %d,A[%d][%d]:%0.12f ,sum %0.12f\n",my_rank,i,j,my_rank*chunk+ i,j,A[my_rank*chunk+ i][j],sum);
			sub_L[i][j]=A[my_rank*chunk+ i][j] - sum;
		}

		MPI_Gather(sub_L,chunk*n,MPI_DOUBLE,&(L[0][0]),chunk*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		// Checking for L[j][j]==0 once
		if(my_rank==0){
			// printf("L[%d][%d]: %0.12f\n",j,j,L[j][j] );
			if (L[j][j] == 0) {
				printf("-------------------------------------TERMINATED\n");
				MPI_Finalize();
				exit(0);
			} 
		}
		
		MPI_Scatter(&(U[0][0]),chunk*n,MPI_DOUBLE,sub_U,chunk*n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Bcast(L[j],n,MPI_DOUBLE,0,MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		// U's Loop
		for (i=left_i; i<right_i; i++) {
			sum = 0;
			for(k = 0; k < j; k++) {
				sum = sum + L[j][k] * sub_U[i][k];
			} 
			sub_U[i][j] =(A[j][chunk*my_rank+i] - sum) / L[j][j];
		}
		MPI_Gather(sub_U,chunk*n,MPI_DOUBLE,&(U[0][0]),chunk*n,MPI_DOUBLE,0,MPI_COMM_WORLD);

		// Placed barrier at the end, so that all processes complete before going to next loop
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	
	// CROUT MATRIX DECOMPOSITION ENDS HERE

	// Convert double[][] to double** here before calling print_matrix()
	// scatter only works with double[][]
	// if we try double**, then 1. not contiguous, 2. we might place burden on process 0 to scatter data.

	if(my_rank==0){

		double* A1[n],*L1[n],*U1[n];
		for(i=0;i<n;i++){
			A1[i] = (double*)malloc(sizeof(double)*n);
			L1[i] = (double*)malloc(sizeof(double)*n);
			U1[i] = (double*)malloc(sizeof(double)*n);
		}

		for(i=0;i<n;i++){
			for(j=0;j<n;j++){
				A1[i][j]=A[i][j];
				L1[i][j]=L[i][j];
				U1[i][j]=U[j][i];
			}
		}

		printf("A\n");
		// now check
		print_matrix(A1,n);

		printf("------------------------------\n");
		printf("L\n");
		// output_(L/U)_<strategy(0/1/2/3/4)>_<number of threads/processes(2/4/8/16)>.txt
		print_matrix(L1,n);

		char str[100] = "output_L_4_";
		char intbuf[10];
		sprintf(intbuf,"%d",comm_sz);
		strcat(str,intbuf);
		strcat(str,".txt");
		
		write_output(str,L1,n);

		printf("------------------------------\n");
		printf("U\n");
		print_matrix(U1,n);
		char str2[100] = "output_U_4_";
		strcat(str2,intbuf);
		strcat(str2,".txt");
		write_output(str2,U1,n);
	}
	  
	/*    */
	MPI_Finalize();
	return 0;
}