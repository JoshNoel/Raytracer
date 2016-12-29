#pragma once
#include "Plane.h"
#include "CudaDef.h"
#include "glm/glm.hpp"
#include "managed.h"

class Light
	: public Managed
{
public:

	enum LIGHT_TYPE
	{
		POINT = 0,
		DIRECTIONAL
	};

	explicit Light(Plane** areaShape = nullptr, const glm::vec3& pos = glm::vec3(0,0,0), const glm::vec3& color = glm::vec3(255,255,255),
		float intensity = 10.0f, LIGHT_TYPE type = LIGHT_TYPE::POINT);

	explicit Light(Plane** areaShape);

	Light(const Light&);
	~Light();
	Light& operator=(const Light&) = delete;
	Light& operator=(Light&&) = delete;

	//calculates direction of light from x, y, and z angles
	CUDA_DEVICE CUDA_HOST void calcDirection(float xAngle, float yAngle, float zAngle);

	//creates an area light shape from a plane
	void setShape(Plane** shape) { host_areaShape = shape;  }
	void finalize() 
	{ 
		if(host_areaShape)
			areaShape = *host_areaShape; 
	}

	LIGHT_TYPE type;
	float intensity;
	glm::vec3 color;
	glm::vec3 pos;
	glm::vec3 dir = glm::vec3(0,0,-1);
	bool castsShadow = true;
	bool isAreaLight = false;

	//device pointer
	Plane* areaShape;
	Plane** host_areaShape;
};

