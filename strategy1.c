#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

void summation_matrix_L(double** L,double** U,double &sum,int i,int j){
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
		sum = sum + local_sum;
	}
}

void summation_matrix_U(double** L,double** U,double &sum,int i,int j){
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
		sum = sum + local_sum;
	}
}

void crout(double const **A, double **L, double **U, int n) {
	int i, j, k;
	double sum = 0;
	#pragma omp parallel for private(i) shared(U,n) schedule(static)
	{
		for (i = 0; i < n; i++) {
			U[i][i] = 1;
		} 
	}

	// implicit barrier

	/*
		Three nested loops and 2 threads in each would lead to 2^3 = 8 threads in total
		And, we are given atleast 8 cores.
	*/

	#pragma omp parallel for private(i,j,k,sum) shared(A,L,U,n) schedule(dynamic) num_threads(2)
	{
		for (j=0; j<n; j++) {
			#pragma omp parallel for private(i,k,sum) shared(j,A,L,U,n) schedule(static) num_threads(2)
			{
				for (i = j; i < n; i++) {
					sum = 0;
					#pragma omp parallel num_threads(2) shared(i,j,L,U,sum)
					{
						summation_matrix_L(L,U,sum,i,j);
					}
					L[i][j] = A[i][j] -	sum;
				}
			}
			#pragma omp parallel for private(i,k,sum) shared(j,A,L,U,n) schedule(static) num_threads(2)
			{
				for (i=j; i<n; i++) {
					sum = 0;
					#pragma omp parallel num_threads(2) shared(i,j,L,U,sum)
					{
						summation_matrix_U(L,U,sum,i,j);
					}
					if (L[j][j] == 0) {
						exit(0);
					} 
					U[j][i] =(A[j][i] -	sum) / L[j][j];
				}
			}
		}
	}
}

int int main(int argc, char** argv[])
{
	/*
		Strategy 1 almost complete, now simply need to take input and test for both strategy 1 and 0.
		On terminal "export OMP_NESTED=TRUE", to be included in run.sh

	*/
	return 0;
}