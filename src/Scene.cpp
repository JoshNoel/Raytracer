#include "Scene.h"
#include "Node.h"

Scene::Scene()
{
#ifdef USE_CUDA
	gpuData = new GpuData();
#endif
    SQRT_DIV2_SHADOW_SAMPLES = std::sqrt(SHADOW_SAMPLES) / 2.0f;
	if (SQRT_DIV2_SHADOW_SAMPLES < 1)
		SQRT_DIV2_SHADOW_SAMPLES = 1;
}

Scene::~Scene()
{
#ifdef USE_CUDA
	CUDA_CHECK_ERROR(cudaFree(gpuData->objectList));
	CUDA_CHECK_ERROR(cudaFree(gpuData->lightList));
	delete gpuData;
#endif
}