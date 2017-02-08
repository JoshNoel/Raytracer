#ifndef CUDA_DEFINES_H
#define CUDA_DEFINES_H


//TODO: Create custom vector class that can use unified memory

//consolidate cuda includes for easy exclusion
#ifdef USE_CUDA
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "curand.h"
#include "curand_kernel.h"
#include "curand_uniform.h"
#include "helper_cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <host_defines.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>
#include <intrin.h>
#endif
#include <vector>

//#define CUDA_CHECK_ERROR(error) cudaErrorCheck(error, __FILE__, __LINE__);
#define CUDA_CHECK_ERROR(error)			\
{										\
	checkCudaErrors((error));			\
}

template<class T>
using vector = std::vector<T>;

/*
#ifdef USE_CUDA
	template<class T>
	using vector = thrust::host_vector<T>;
#else
	template<class T>
	using vector = std::vector<T>;
#endif
*/
/*
static inline void cudaErrorCheck(cudaError_t error, const char* file, unsigned line) {
	if(error != cudaSuccess) {
		std::string errorString = "Cuda Error: "  + std::string(cudaGetErrorString(error) + std::string(file) + std::string(line));
		throw std::runtime_error(errorString);
	}
}*/
#if defined(USE_CUDA)
	#define BLOCK_SIZE 1024
#endif
#if defined(USE_CUDA)
	#define CUDA_CONST __constant__
	#else
	#define CUDA_CONST

#endif

#if defined(USE_CUDA)
	#define CUDA_HOST __host__
	#else
	#define CUDA_HOST
#endif

#if defined(USE_CUDA)
	#define CUDA_DEVICE __device__
	#else
	#define CUDA_DEVICE
#endif

#if defined(USE_CUDA)
	#define CUDA_GLOBAL __global__
	#else
	#define CUDA_GLOBAL
#endif


#if defined(USE_CUDA) && defined(__CUDACC__)
#define CUDA_LAUNCH_BOUNDS(max_reg_usage, min_block_size) __launch_bounds__(max_reg_usage, min_block_size)
#else
#define CUDA_LAUNCH_BOUNDS(max_reg_usage, min_block_size)
#endif


#if defined(USE_CUDA) && defined(__CUDACC__)
#define KERNEL_ARGS2(kernel_dim, block_dim) <<< kernel_dim, block_dim >>>
#else
	#define KERNEL_ARGS2(kernel_dim, block_dim)
#endif

#if defined(USE_CUDA) && defined(__CUDACC__)
#define KERNEL_ARGS4(kernel_dim, block_dim, shared_mem, stream) <<< kernel_dim, block_dim, shared_mem, stream >>>
#else
#define KERNEL_ARGS4(kernel_dim, block_dim)
#endif

#endif
