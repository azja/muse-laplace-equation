#include "../laplace.h"
////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h> 
#include <stdlib.h>
#include <boost/progress.hpp>
#include <ostream>

////////////////////////////////////////////////////////////////////////////////////////

void ERROR_HANDLER(cudaError_t err)
{
    if(err != cudaSuccess)
	{
	printf("Error code: %d:%s ...exiting",err,cudaGetErrorString( err ));
	exit(1);
	}
}

////////////////////////////////////////////////////////////////////////////////////////

const float PI = 3.14159;
void filler(float* M, uint ROWS,uint COLUMNS)
{
	for(int i = 0;i < ROWS;++i){
		for(int j = 0; j< COLUMNS;++j)
		{
		if( 0 != i)
		M[i*ROWS + j] = 0.0f;
		else
		M[j] = 30.0f * sin(4*PI*float(j)/COLUMNS)+30.0f;
		
		if(j == COLUMNS-1 || j == 0)
		
		M[i*ROWS + j] = 30.0f;
		}
	}
	
	
}

void filler1(float* M, uint ROWS,uint COLUMNS)
{
	for(int i = 0;i < ROWS;++i){
		for(int j = 0; j< COLUMNS;++j)
		{
		    M[i*ROWS +j] = -1;
		}
	}


	for(int i = 0;i < ROWS;++i){
		for(int j = 0; j< COLUMNS;++j)
		{
		    if((i-ROWS/2)*(i-ROWS/2) + (j-COLUMNS/2)*(j-COLUMNS/2) < ROWS*ROWS * 0.25)
		    M[i*ROWS +j ] = 0.0f;
		}
	}
	
	M[ROWS*ROWS/2 + COLUMNS/2] = -1.0;
	M[ROWS*ROWS/2 + COLUMNS/2-1] = 200.0f;
	M[ROWS*ROWS/2 + COLUMNS/2+1] = 200.0f;
	M[ROWS*ROWS/2 + COLUMNS/2 + ROWS] = 200.0f;
	M[ROWS*ROWS/2 + COLUMNS/2-ROWS] = 200.0f;


	
	
}


////////////////////////////////////////////////////////////////////////////////////////

typedef  float type;

const uint nThreads = 512;
const uint matrixDimX = 2  *  nThreads;
const uint matrixDimY =  2 *  nThreads;
const uint nBlocks = matrixDimX / nThreads;
const uint matrixDimHalo = matrixDimX + 2;

const uint  sizeN = sizeof(float) * (matrixDimHalo) *(matrixDimHalo);

const uint N = 100000;

typedef void (*func)(float*,uint,uint);




int main(int argc,char* argv[])
{


type  *d_in;
type  *d_out;

func fill = filler;

ERROR_HANDLER(cudaMalloc((void**)&d_in,sizeN));
ERROR_HANDLER(cudaMalloc((void**)&d_out,sizeN));

type* input = new float[matrixDimHalo * matrixDimHalo];
type* output = new float[matrixDimHalo * matrixDimHalo];



fill(input,matrixDimHalo,matrixDimHalo);
fill(output,matrixDimHalo,matrixDimHalo);


ERROR_HANDLER(cudaMemcpy(d_in,input,sizeN,cudaMemcpyHostToDevice));
ERROR_HANDLER(cudaMemcpy(d_out,output,sizeN,cudaMemcpyHostToDevice));



type *temp;
FILE *f;
if(argc > 1)
{
	char *filename =  argv[1];
	f = fopen(filename,"w");
}
else
f = fopen("wynik.txt","w");




 boost::progress_display show_progress(N);

for(int i = 0;i < N;++i)
{
museCudaLaplace<float><<<nBlocks,nThreads,(nThreads+2) * 3 * sizeof(float)>>>(d_in,d_out,nThreads,matrixDimX,matrixDimY);
cudaThreadSynchronize();
temp = d_out;
d_out = d_in;
d_in = temp;
 ++show_progress; //boost progress indicator
}


ERROR_HANDLER(cudaMemcpy(output,d_out,sizeN,cudaMemcpyDeviceToHost));
ERROR_HANDLER(cudaMemcpy(input,d_in,sizeN,cudaMemcpyDeviceToHost));

for(int i = 0; i < matrixDimHalo; ++i){
	for(int j = 0; j < matrixDimHalo;++j)
		if( input[i*matrixDimHalo + j] >= 0.0f)
			fprintf(f,"%d %d %f \n",i,j,input[i*matrixDimHalo + j]);
		fprintf(f,"\n");
	}

ERROR_HANDLER(cudaFree(d_in));
ERROR_HANDLER(cudaFree(d_out));

delete [] input;
delete [] output;

fclose(f);

return 0;
}



