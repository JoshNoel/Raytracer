#pragma once
#include "CudaDef.h"
#include "Ray.h"
#include "Sphere.h"
#include "Image.h"
#include "Camera.h"
#include "Scene.h"
#include <random>
#include <atomic>


class Renderer : public Managed
{
public:
	//ensure these pointers are to heap in order for cuda unified memory to function
	Renderer(Scene* scene, Image* image, Camera* cam);

	bool init();

#ifdef USE_CUDA
	//const to ensure threads can call
	CUDA_HOST void renderKernel(dim3 kernelDim, dim3 blockDim, curandState_t* states);
	//launches kernel
	void renderCuda();
#endif

#ifndef USE_CUDA
	glm::vec3 renderPixel(int x , int y) const;
#endif
	void writeImage(glm::vec3 color, int x, int y) const;


	Image* image;
	Camera* camera;

	Scene* scene;

#ifdef USE_CUDA
	Scene::GpuData* sceneGpuData;
	curandState_t* states;
#endif

	//casts ray into the scene and returns color at intersection point if there is an intersection
		//thit0 and thit1 are inputs that are set to the t-values representing the intersection point on the ray
		//depth represents what order ray is being cast (i.e. primary, secondary, tertiary, etc.)
	CUDA_DEVICE glm::vec3 castRay(Ray& ray, float& thit0, float& thit1, int depth) const;
	CUDA_DEVICE glm::vec3 castRay(Ray& ray, int depth) const;

	//returns whether a ray hits an object
	CUDA_DEVICE bool hitsObject(Ray& ray, float& thit0, float& thit1) const;
	CUDA_DEVICE bool hitsObject(Ray& ray) const;

	//used to generate random points on an area light to crate soft shadows
	mutable std::minstd_rand rng;
	mutable std::uniform_real_distribution<float> distributionX;
	mutable std::uniform_real_distribution<float> distributionY;

	static const float SHADOW_RAY_LENGTH;
	static const int NUM_THREADS;

	mutable std::atomic<int> pixelsRendered;

	//number of threads per block in each dimension (2D)
	//calculate # of blocks in image dims using these values
	int BLOCK_DIM_X, BLOCK_DIM_Y;
	int NUM_BLOCKS_X, NUM_BLOCKS_Y;
	const int DEFAULT_NUM_BLOCKS = 8;
	const int MAX_BLOCKS = 1024;

	bool extraBlock;

	//determined through profiling (max_registers_per_block / registers_per_thread)
	#define MAX_THREADS_PER_BLOCK 640
	#define MIN_BLOCKS_PER_MULTIPROCESSOR 1
};

