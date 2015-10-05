#pragma once
#include "glm\glm.hpp"
class Material
{
public:
	Material(glm::vec3 col, float ior = 1.0f, float dc= 1.0f);
	Material();
	~Material();

	//calculates unpolarized reflectivity of material from Index of Refraction 1(n1) into the material given the angle of incidence
	// default incoming index of refraction of 1.0f represents air
	float calcReflectivity(float angleOfIncidence, float n1 = 1.0f);

	glm::vec3 color;
	float diffuseCoef;
	float indexOfRefrac;
};

