#ifndef RAYTRACER_CORE_H
#define RAYTRACER_CORE_H

#include "ThreadPool.h"
#include "Renderer.h"
#include "CudaLoader.h"

class Core
{
public:
	Core(Renderer*);
	~Core();

	void render();
	void setCudaLoader(CudaLoader* c) { this->cudaLoader = c; }
private:
	ThreadPool threadPool;
	Renderer* renderer;
	CudaLoader* cudaLoader;

};

#endif
