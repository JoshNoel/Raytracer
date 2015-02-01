#pragma once
#include "glm\glm.hpp"

class Light
{
public:
	Light();
	Light(glm::vec3 p, glm::vec3 c, float i);
	~Light();
	enum LIGHT_TYPE
	{
		POINT = 0
	};

	float intensity;
	glm::vec3 color;
	glm::vec3 pos;
};

