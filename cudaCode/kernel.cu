#include "../external/GameCuda.h"


__device__ __forceinline__
uint8_t Alive(int number)
{
	if (number == 765|| // if true the sum is either 3 alive or 2 alive
		number == 510)  // and i make the pixel black
	{
		return 255; 
	}
	return 0;
}
__global__
void Checker(uint8_t* __restrict__ d_Data, uint8_t* result, int width, int height) // what a name for a function, right? ha-ha ,it was a long process until i name it that way
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	int threadId = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
	if (x >= width || x < 0
		|| y >= height || y < 0)
	{
		return;
	}
	if (x>width || x<1 ||y>height || y<1) // the special case with broders 
	{
		result[threadId] = 255;
		return;
	}
	int sum = // i know it is a global read, but imo the first memory load may load everything needed
		d_Data[threadId - width - 1] +
		d_Data[threadId - 1] +
		d_Data[threadId + 1] +
		d_Data[threadId - width] +
		d_Data[threadId + width] +
		d_Data[threadId + width + 1] +
		d_Data[threadId + width - 1] +
		d_Data[threadId - width + 1];


	uint8_t aliveOrNot = d_Data[threadId];
	if (aliveOrNot == 0)//well this is kind of fucked up, because i did not want to use if statements
						//but this is the only solution i though of
	{
		result[threadId] = Alive(sum);
		return;
	}
	if (sum = 765)
	{
		result[threadId] = 255;
		return;
	}
	result[threadId] = 0;
}

void CudaGame::kernel(cudaStream_t& providedstream)
{
	dim3 sizeOfBlock(((width + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK), height);

	Checker << <sizeOfBlock, THREADS_PER_BLOCK, 0, providedstream >> > (d_image, d_result, width, height);
	auto status = cudaGetLastError();
}
