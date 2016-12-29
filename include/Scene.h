#pragma once
#include <memory>
#include "Light.h"
#include "CudaDef.h"
#include "glm/glm.hpp"
#include "GeometryObj.h"
#include "Node.h"

class Scene
{
public:
	Scene();
	~Scene();

	struct GpuData : public Managed {
		//copy vectors to arrays for use on gpu
		GeometryObj** objectList;
		Light** lightList;

		unsigned objectListSize;
		unsigned lightListSize;

		//need to copy bgColor, ambientColor, ambientIntensity to constant memory
		glm::vec3 ambientColor;
		float ambientIntensity;
		glm::vec3 bgColor;

		int SQRT_DIV2_SHADOW_SAMPLES;
	};

	//adds an object to the scene and sets its id
	void addObject(GeometryObj* o)
	{ 
		o->id = idCounter++;
		objectList.push_back(o);
	}


	//adds a light to the scene
	void addLight(Light* l){ lightList.push_back(l); }

	//sets ambient color of the scene
		//minimum color for a point in shadow
	void setAmbient(const glm::vec3& color, float intensity)
	{
		ambientColor = color;
		ambientIntensity = intensity;
	}

	//sets background color of the scene
	void setBgColor(const glm::vec3& col)
	{
		bgColor = col;
	}

	void finalizeCUDA() {
#ifdef USE_CUDA
		//fill structure with GPU specific data

		//copies geometryObj pointer vector to device
		gpuData->SQRT_DIV2_SHADOW_SAMPLES = Scene::SQRT_DIV2_SHADOW_SAMPLES;
		gpuData->ambientColor = ambientColor;
		gpuData->ambientIntensity = ambientIntensity;
		gpuData->bgColor = bgColor;
		gpuData->lightListSize = lightList.size();
		gpuData->objectListSize = objectList.size();
		CUDA_CHECK_ERROR(cudaMalloc((void**)&gpuData->objectList, objectList.size() * sizeof(GeometryObj)));
		CUDA_CHECK_ERROR(cudaMalloc((void**)&gpuData->lightList, lightList.size() * sizeof(Light)));
		CUDA_CHECK_ERROR(cudaMemcpy(gpuData->objectList, objectList.data(), objectList.size() * sizeof(GeometryObj), cudaMemcpyHostToDevice));
		CUDA_CHECK_ERROR(cudaMemcpy(gpuData->lightList, lightList.data(), lightList.size() * sizeof(Light), cudaMemcpyHostToDevice));
#endif
	}

	//__CUDA_ARCH__ is defined if code is running on gpu
	CUDA_HOST CUDA_DEVICE const glm::vec3 getAmbientColor() const {
	#if (defined(__CUDA_ARCH__)) && (__CUDA_ARCH__ > 0)
			return gpuData->ambientColor;
	#else
			return this->ambientColor;
	#endif
		}

	CUDA_HOST CUDA_DEVICE float getAmbientIntensity() const {
	#if (defined(__CUDA_ARCH__)) && (__CUDA_ARCH__ > 0)
				return gpuData->ambientIntensity;
	#else
				return this->ambientIntensity;
	#endif
	}

	CUDA_HOST CUDA_DEVICE const glm::vec3 getBgColor() const {
	#if (defined(__CUDA_ARCH__)) && (__CUDA_ARCH__ > 0)
				return gpuData->bgColor;
	#else
				return this->bgColor;
	#endif
	}



#ifdef USE_CUDA

	CUDA_HOST CUDA_DEVICE GpuData* getGpuData() const {
		return gpuData;
	}

	CUDA_HOST CUDA_DEVICE GeometryObj** getDeviceObjectList() const{
		return gpuData->objectList;
	}


	CUDA_HOST CUDA_DEVICE Light** getDeviceLightList() const{
		return gpuData->lightList;
	}

	CUDA_HOST CUDA_DEVICE unsigned int getObjectListSize() const {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
		return gpuData->objectListSize;
#else
		return objectList.size();
#endif
	}

	CUDA_HOST CUDA_DEVICE unsigned int getLightListSize() const {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
		return gpuData->lightListSize;
#else
		return lightList.size();
#endif
	}
#endif

	const vector<GeometryObj*>& getObjectList() const {
		return this->objectList;
	}


	const vector<Light*>& getLightList() const {
		return this->lightList;
	}


	int idCounter = 0;

	//static const int host variable (and float if not on windows) can be accessed directly by device
	static const int MAX_RECURSION_DEPTH = 2;

	//samples to cast to an area light to determine shadow intensity
		//if a ray doesn't intersect an object on the way to the light, the point is lit
		//if it does intersect it is in shadow
		//average the results of the samples to determine visibililty of point to an area light
			//1 = completly visible
			//0 = completly in shadow
	//static const int host variable (and float if not on windows) can be accessed directly by device
	static const int SHADOW_SAMPLES = 16;
	int SQRT_DIV2_SHADOW_SAMPLES;

	//samples to cast per pixel
		//also uses stratified random sampling (like with the area lights) within each pixel
		//average color of the primary rays to get final color of the pixel
	//static const int host variable (and float if not on windows) can be accessed directly by device
	static const int PRIMARY_SAMPLES = 1;



private:

	//Axis-aligned bounding box for the scene as a whole
	BoundingBox sceneBox;


	glm::vec3 ambientColor = glm::vec3(255,255,255);
	float ambientIntensity = 0.01f;
	glm::vec3 bgColor = glm::vec3(0,0,0);

	//store objects and lights that are a part of the scene
	vector<GeometryObj*> objectList;
	vector<Light*> lightList;

#ifdef USE_CUDA
	GpuData* gpuData;
#endif
};
