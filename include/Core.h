#ifndef RAYTRACER_CORE_H
#define RAYTRACER_CORE_H

#include "ThreadPool.h"
#include "Renderer.h"

class Core
{
public:
	Core(const Renderer*);
	~Core();

	void render();

private:
	ThreadPool threadPool;
	const Renderer* renderer;
};

#endif