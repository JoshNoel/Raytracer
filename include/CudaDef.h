#ifndef CUDA_DEFINES_H
#define CUDA_DEFINES_H

#define CUDA_CHECK_ERROR(error) cudaErrorCheck(error, __FILE__, __LINE__);

static inline void cudaErrorCheck(cudaError_t error, const char* file, unsigned line) {
	if(error != cudaSuccess) {
		std::string errorString = "Cuda Error: "  + std::string(cudaGetErrorString(error) + std::string(file) + std::string(line));
		throw std::runtime_error(errorString);
	}
}

#ifdef USE_CUDA
	#define CUDA_CONST __constant__
	#else
	#define CUDA_CONST
#endif

#undef CREATE_CUDA_DEF

#endif
