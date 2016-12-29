#pragma once
#include "CudaDef.h"
#include "glm/glm.hpp"

class GeometryObj;
class Triangle;
class Ray
{
	/*
	*Ray = position + direction (t)
	*/
public:
	CUDA_HOST CUDA_DEVICE Ray();
	CUDA_HOST CUDA_DEVICE Ray(glm::vec3, glm::vec3);
	CUDA_HOST CUDA_DEVICE ~Ray();

	glm::vec3 pos;
	glm::vec3 dir;

	//hit closer to origin of ray
	float thit0;

	//hit farther from origin of ray
	float thit1;
	//thit0 = thit1 if there is only 1 intersection

	//object ray has intersected
	GeometryObj* hitObject;

	//triangle ray has intersected, if it intersects a TriObject
	Triangle* hitTri;
};

