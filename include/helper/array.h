#ifndef RAYTRACER_ARRAY_H
#define RAYTRACER_ARRAY_H
#include "managed.h"
#include <memory>
#include "CudaDef.h"

namespace helper
{

template<typename T, unsigned N, bool managed = false>
class array {
public:
	CUDA_HOST CUDA_DEVICE array()
	{
		vals = new T[N];
		size = N;
	}

	CUDA_HOST CUDA_DEVICE array(const array<T, N, false>& a)
	{
		vals = new T[N];
		size = N;

		//deep copy array
		for(unsigned i = 0; i < N; i++)
		{
			vals[i] = a[i];
		}
	}


	CUDA_HOST CUDA_DEVICE array(const array<T, N, true>& a)
	{
		vals = new T[N];
		size = N;

		//deep copy array
		for (unsigned i = 0; i < N; i++)
		{
			vals[i] = a[i];
		}
	}


	CUDA_HOST CUDA_DEVICE array(array<T, N, false>&& a)
	{
		vals = a.vals;
		a.vals = nullptr;
		size = N;
	}

	CUDA_HOST CUDA_DEVICE array<T, N>& operator=(const array<T, N, false>& a)
	{
		for(unsigned i = 0; i < N; i++)
		{
			vals[i] = a[i];
		}

		return *this;
	}


	CUDA_HOST CUDA_DEVICE array<T, N>& operator=(const array<T, N, true>& a) = delete;

	CUDA_HOST CUDA_DEVICE array<T, N>& operator=(array<T, N, true>&& a) = delete;

	CUDA_HOST CUDA_DEVICE array<T, N>& operator=(array<T, N>&& a)
	{
		vals = a.vals;
		a.vals = nullptr;

		return *this;
	}

	CUDA_HOST CUDA_DEVICE ~array()
	{
		delete[] vals;
	}

	CUDA_HOST CUDA_DEVICE T& operator[](const int i) {
		return vals[i];
	}

	CUDA_HOST CUDA_DEVICE const T& operator[](const int i) const{
		return vals[i];
	}

	CUDA_HOST CUDA_DEVICE T* data() { return &vals[0]; }

	CUDA_HOST CUDA_DEVICE operator array<T, N, true>() { return *this; }

private:
	T* vals;
	size_t size;
};

template<typename T, unsigned N>
class array<T,N,true> : public Managed{
public:
	CUDA_HOST CUDA_DEVICE array()
	{
		vals = static_cast<T*>(Managed::operator new[](sizeof(T)*N));

		size = N;

		//need to manually initialize values as operator new does not do this
		for(auto i = 0; i < N; i++)
		{
			T temp = T();
			vals[i] = temp;
		}
	}

	CUDA_HOST array(const array<T, N, false>& a)
	{
		vals = static_cast<T*>(Managed::operator new[](sizeof(T)*N));
		size = N;

		//deep copy array
		for (unsigned i = 0; i < N; i++)
		{
			vals[i] = a[i];
		}
	}

	CUDA_HOST array(const array<T, N, true>& a)
	{
		vals = static_cast<T*>(Managed::operator new[](sizeof(T)*N));
		size = N;

		//deep copy array
		for (unsigned i = 0; i < N; i++)
		{
			vals[i] = a[i];
		}
	}

	CUDA_HOST array(array<T, N, true>&& a)
	{
		vals = a.vals;
		a.vals = nullptr;
		size = N;
	}

	CUDA_HOST CUDA_DEVICE array<T, N, true>& operator=(const array<T, N, false>& a)
	{
		for (unsigned i = 0; i < N; i++)
		{
			vals[i] = a[i];
		}

		return *this;
	}

	CUDA_HOST CUDA_DEVICE array<T, N, true>& operator=(const array<T, N, true>& a)
	{
		for (unsigned i = 0; i < N; i++)
		{
			vals[i] = a[i];
		}

		return *this;
	}

	CUDA_HOST CUDA_DEVICE array<T, N, true>& operator=(array<T, N, true>&& a)
	{
		vals = a.vals;
		a.vals = nullptr;

		return *this;
	}


	CUDA_HOST CUDA_DEVICE ~array()
	{
		//need to manually call destructor because we are using operator delete[]
		//can call destructor even if T happens to be builtin type
			//http://stackoverflow.com/questions/456310/destructors-of-builtin-types-int-char-etc
		for(auto i = 0; i < N; i++)
		{
			vals[i].~T();
		}
		Managed::operator delete[](vals);
	}

	CUDA_HOST CUDA_DEVICE T& operator[](const int i) {
		return vals[i];
	}

	CUDA_HOST CUDA_DEVICE const T& operator[](const int i) const {
		return vals[i];
	}

	CUDA_HOST CUDA_DEVICE T* data() { return &vals[0]; }

private:
	T* vals;
	size_t size;
};
}



#endif
