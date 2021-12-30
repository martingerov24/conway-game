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
	CudaGame(std::vector<uint8_t>& img,
		const int height, const int width)

		: img(img), height(height)
		, width(width), d_image(nullptr)
	{
		cudaStatus = cudaError_t(0); // this does not have to be deleted in deconstructor
	}

	__host__ void streamSetupAndCreate()
	{
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		cudaStatus = cudaSetDevice(0);
		assert(cudaStatus == cudaSuccess && "you do not have cuda capable device!");
	}
	__host__ void memoryAllocationAsyncOnDevice(cudaStream_t& providedstream)
	{
		cudaStatus = cudaMallocAsync((void**)&d_image, sizeof(uint8_t) * img.size(), providedstream);
		assert(cudaStatus == cudaSuccess && "not able to allocate memory on device1");
	}

	__host__ void cudaUploadImage(cudaStream_t& providedstream)
	{
		cudaStatus = cudaMemcpyAsync(d_image, img.data(), sizeof(uint8_t) * img.size(), cudaMemcpyHostToDevice, providedstream);
		assert(cudaStatus == cudaSuccess && "not able to tansfer Data!");
	}
	__host__ void kernel(cudaStream_t& providedstream);
	__host__ void downloadAsync(cudaStream_t& provided_stream, std::vector<uint8_t>& result, int size)
	{
		result.resize(size);
		cudaStatus = cudaMemcpyAsync(result.data(), d_image, sizeof(uint8_t) * size, cudaMemcpyDeviceToHost, provided_stream);
		assert(cudaStatus == cudaSuccess && "download Async");
	}

	__host__ void sync(cudaStream_t& providedStream)
	{
		cudaStatus = cudaStreamSynchronize(providedStream);
	}

	__host__ void cudaFreeAcyncMatcher(cudaStream_t& provided_stream)
	{
		cudaStatus = cudaFreeAsync(d_image, provided_stream);
		assert(cudaStatus == cudaSuccess && "cuda free async");
	}
	__host__ ~CudaGame()
	{
	}
private:
	//-------Provided By Main-------
	std::vector<uint8_t>& img;
	const int height, width;
	//------Provided By Class,------
	//should be deleted after using
	uint8_t* d_image;
	cudaError_t cudaStatus;
};
