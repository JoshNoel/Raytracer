#ifndef CUDA_DEFINES_H
#define CUDA_DEFINES_H

#define CUDA_CHECK_ERROR(error) cudaErrorCheck(error, __FILE__, __LINE__);

static inline void cudaErrorCheck(cudaError_t error, const char* file, unsigned line) {
	if(error != cudaSuccess) {
		std::string errorString = "Cuda Error: "  + std::string(cudaGetErrorString(error) + std::string(file) + std::string(line));
		throw std::runtime_error(errorString);
	}
}
#if defined(USE_CUDA) && !defined(BLOCK_SIZE)
	#define BLOCK_SIZE 1024
#endif
#if defined(USE_CUDA) && !defined(CUDA_CONST)
	#define CUDA_CONST __constant__
	#else
	#define CUDA_CONST

#endif

#if defined(USE_CUDA) && !defined(CUDA_HOST)
	#define CUDA_HOST __host__
	#else
	#define CUDA_HOST
#endif

#if defined(USE_CUDA) && !defined(CUDA_DEVICE)
	#define CUDA_DEVICE __device__
	#else
	#define CUDA_DEVICE
#endif

#if defined(USE_CUDA) && !defined(CUDA_GLOBAL)
	#define CUDA_GLOBAL __constant__
	#else
	#define CUDA_GLOBAL
#endif

#endif
