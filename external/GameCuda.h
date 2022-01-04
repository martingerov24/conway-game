#pragma once
#define THREADS_PER_BLOCK 1024
#include "cuda_runtime.h"
#include "cuda/std/cmath"
#include "device_launch_parameters.h"
#include <vector>
#include <cassert>

class CudaGame
{
public:
	CudaGame(const std::vector<uint8_t>& img,
		const int height, const int width)

		: img(img), height(height)
		, width(width), d_image(nullptr)
		,d_result(nullptr)
	{
	}
	__host__ void memoryAllocationAsyncOnDevice(cudaStream_t& providedstream, cudaError_t cudaStatus)
	{
		cudaStatus = cudaMallocAsync((void**)&d_image, sizeof(uint8_t) * img.size(), providedstream);
		assert(cudaStatus == cudaSuccess && "not able to allocate memory on device0");

		cudaStatus = cudaMallocAsync((void**)&d_result, sizeof(uint8_t) * img.size(), providedstream);
		assert(cudaStatus == cudaSuccess && "not able to allocate memory on device0, the result memeory");
	}

	__host__ void cudaUploadImage(cudaStream_t& providedstream, cudaError_t cudaStatus)
	{
		cudaStatus = cudaMemcpyAsync(d_image, img.data(), sizeof(uint8_t) * img.size(), cudaMemcpyHostToDevice, providedstream);
		assert(cudaStatus == cudaSuccess && "not able to tansfer Data!");
	}

	__host__ void kernel(cudaStream_t& providedstream);

	__host__ void downloadAsync(cudaStream_t& provided_stream, std::vector<uint8_t>& result, int size, cudaError_t cudaStatus)
	{
		cudaStatus = cudaMemcpyAsync(result.data(), d_result, sizeof(uint8_t) * size, cudaMemcpyDeviceToHost, provided_stream);
		assert(cudaStatus == cudaSuccess && "download Async");
	}

	__host__ void sync(cudaStream_t& providedStream, cudaError_t cudaStatus)
	{
		cudaStatus = cudaStreamSynchronize(providedStream);
	}

	__host__ void cudaFreeAcync(cudaStream_t& provided_stream, cudaError_t cudaStatus)
	{
		cudaStatus = cudaFreeAsync(d_image, provided_stream);
		assert(cudaStatus == cudaSuccess && "cuda free async");

		cudaStatus = cudaFreeAsync(d_result, provided_stream);
		assert(cudaStatus == cudaSuccess && "cuda free async result");
	}
	__host__ ~CudaGame()
	{
	}
private:
	//-------Provided By Main-------
	const std::vector<uint8_t>& img;
	const int height, width;
	//------Provided By Class,------
	//should be deleted after using
	uint8_t* d_image;
	uint8_t* d_result;
};
