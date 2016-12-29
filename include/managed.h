#ifndef RAYTRACER_MANAGED_H
#define RAYTRACER_MANAGED_H

#include "CudaDef.h"
#include <new>
#include <vector>

class Managed {
public:
	CUDA_HOST CUDA_DEVICE void* operator new(size_t size) {
#if defined(USE_CUDA) && !((defined(__CUDA_ARCH__)) && (__CUDA_ARCH__ > 0))

		void* p;
		check((cudaMallocManaged(&p, size)), "", __FILE__, __LINE__);
		//CUDA_CHECK_ERROR(cudaMallocManaged(&p, size));
		CUDA_CHECK_ERROR(cudaDeviceSynchronize());
		return p;
#else
		return ::operator new(size);
#endif
	}

	CUDA_HOST CUDA_DEVICE void* operator new[](size_t size) {
#if defined(USE_CUDA) && !((defined(__CUDA_ARCH__)) && (__CUDA_ARCH__ > 0))

		void* p;
		check((cudaMallocManaged(&p, size)), "", __FILE__, __LINE__);
		//CUDA_CHECK_ERROR(cudaMallocManaged(&p, size));
		CUDA_CHECK_ERROR(cudaDeviceSynchronize());
		return p;
#else
		return ::operator new(size);
#endif
	}

	CUDA_HOST CUDA_DEVICE void operator delete(void* p) {
#if defined(USE_CUDA) && !((defined(__CUDA_ARCH__)) && (__CUDA_ARCH__ > 0))
		CUDA_CHECK_ERROR(cudaDeviceSynchronize());
		check((cudaFree(p)), "", __FILE__, __LINE__);

		//CUDA_CHECK_ERROR(cudaFree(p));
#else
		::operator delete(p);
#endif
	}


	CUDA_HOST CUDA_DEVICE void operator delete[](void* p) {
#if defined(USE_CUDA) && !((defined(__CUDA_ARCH__)) && (__CUDA_ARCH__ > 0))
		CUDA_CHECK_ERROR(cudaDeviceSynchronize());
		check((cudaFree(p)), "", __FILE__, __LINE__);

		//CUDA_CHECK_ERROR(cudaFree(p));
#else
		::operator delete(p);
#endif
	}
};


#endif
