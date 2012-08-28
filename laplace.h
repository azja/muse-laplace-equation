/*! \file  laplace.h 
    \brief This file contains template method: 2d  Laplace equation solver engine for  NVidia CUDA supported device
    
    Part of MuSE library - merging with other codes to homogenous namespace structure is  planned in the nearest future
*/
#include <cuda_runtime.h>


typedef unsigned int uint;


/*! \fn template<typename T> museCudaLaplace(T *A,T *B,int BLOCK_SIZE,int COLUMN_SIZE,int ROW_SIZE)
    \brief Sweeps given 2d matrix performing Jacobi procedure for Laplace equation solving.
    \tparam T floating type (double precision if supported on device )
    \param  *A input flatten 2d matrix of size (\a ROW_SIZE + 2) x (\a COLUMN_SIZE + 2)
    \param  *B output flatten 2d matrix of size (\a ROW_SIZE + 2) x (\a COLUMN_SIZE + 2)
    \param  BLOCK_SIZE number of threads in single block
    \param COLUMN_SIZE number of columns for a given grid
    \param ROW_SIZE number of rows for a given grid
    
    Input matrix define geometry and bonduary condtions for a given 2d sample:
     <pre>
     - A[0][i], A[\a ROW_SIZE + 1][i], A[i][0],A[i][\a COLUMN_SIZE + 1] cells stores values related to bonduary conditions
     - \a ROW_SIZE has to fulfill relation: \a ROW_SIZE mod (\a BLOCK_SIZE) = 0
     - Each 2d mappable shape may be considered by introducing "-1" value for cell, each neighbour of "-1" cell becomes external bonduary element e.g.
     
     1   1   1   1   1 \n
     1   x   x   x   1 \n
     1   x   5   x   1 \n
     1   5  -1   5   1 \n
     1   x   5   x   1 \n
     1   x   x   x   1 \n
     1   1   1   1   1 \n
     </pre>
     "x" are cells which will be considered during computation, six of them having neigbours with constant "5" value.
     Procedure sweeps given grid (matrix \a A) once and stores result in \a B - the number of iterations to obtain convergence depends on the size of grid,
     initial values and geometry
     
 */  
template<typename T>
__global__ void museCudaLaplace(T  *A, T *B,int BLOCK_SIZE,int COLUMN_SIZE,int ROW_SIZE)
{

	int RADIUS = 1;
	extern __shared__ 	T internal[];
			
	ROW_SIZE = ROW_SIZE + 2 * RADIUS;
	COLUMN_SIZE = COLUMN_SIZE + 2 * RADIUS;
	
	int BOS = BLOCK_SIZE + 2;
	int COS = 2 * BOS;

	int localId = threadIdx.x + RADIUS;
	int globalId = blockIdx.x * blockDim.x + localId;


	internal[localId] = A[globalId];
	internal[localId + BOS] = A[globalId + ROW_SIZE];
	internal[localId + COS] = A[globalId + 2 * ROW_SIZE ];



	for(int  n = 0;n < COLUMN_SIZE-2; ++n)
	{


		if(threadIdx.x < RADIUS)
			{
			internal[localId - RADIUS] = A[globalId - RADIUS + n * ROW_SIZE];
			internal[localId + BLOCK_SIZE] = A[globalId + BLOCK_SIZE  + n * ROW_SIZE];

			internal[localId - RADIUS + BOS] = A[globalId - RADIUS + (n + 1) * ROW_SIZE];
			internal[localId + BLOCK_SIZE + BOS] = A[globalId + BLOCK_SIZE + (n + 1) * ROW_SIZE];

			internal[localId - RADIUS + COS] = A[globalId - RADIUS +(n + 2) * ROW_SIZE];
			internal[localId + BLOCK_SIZE + COS] = A[globalId + BLOCK_SIZE + (n + 2) * ROW_SIZE];
			}

		__syncthreads();


		T value = T();

		for(int OFFSET = - RADIUS; OFFSET <= RADIUS; ++OFFSET)
			value += internal[BOS + localId + OFFSET];

	
		value = 0.25f*(internal[localId] +internal[localId + COS] + value - internal[BOS + localId]);
	
		__syncthreads();

		internal[localId] =  internal[BOS + localId];
		internal[localId + BOS] =  internal[COS + localId];
		internal[localId + COS] = A[globalId + (n + 3) * ROW_SIZE];

		uint position = (n + 1) * ROW_SIZE + globalId;
		if(A[position] != -1 && A[position - 1] != -1 && A[position + 1] != -1 && A[position - ROW_SIZE] != -1 && A[position + ROW_SIZE] != -1)
			B[position] = value;
	
	}
}


