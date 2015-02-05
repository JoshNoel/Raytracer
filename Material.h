#pragma once
#include "glm\glm.hpp"
class Material
{
public:
	Material(glm::vec3 col, float dc= 1.0f);
	Material();
	~Material();

	glm::vec3 color;
	float diffuseCoef;
};

