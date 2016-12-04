#ifndef RAYTRACER_MANAGED_H
#define RAYTRACER_MANAGED_H
#include <new>
#include "CudaDef.h"

class Managed {
public:
	void* operator new(size_t size) {
#ifdef USE_CUDA
		void* p;
		CUDA_CHECK_ERROR(cudaMallocManaged(&p, size));
		CUDA_CHECK_ERROR(cudaDeviceSynchronize());
		return p;
#else
		return ::operator new(size);
#endif
	}

	void operator delete(void* p) {
#ifdef USE_CUDA
		cudaDeviceSynchronize();
		cudaFree(p);
#else
		::operator delete(p);
#endif
	}
};


#endif
