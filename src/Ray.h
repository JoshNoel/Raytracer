#pragma once
#include "glm\glm.hpp"
class Ray
{
	/*
	*Ray implicit=
	*|pos+t(dir)|
	*
	*
	*/
public:
	Ray();
	Ray(glm::vec3, glm::vec3);
	~Ray();

	glm::vec3 pos;
	glm::vec3 dir;
};

