#pragma once
#include "glm/glm.hpp"
#include "Plane.h"

class Light
{
public:

	enum LIGHT_TYPE
	{
		POINT = 0,
		DIRECTIONAL
	};

	Light(const glm::vec3& pos = glm::vec3(0,0,0), const glm::vec3& color = glm::vec3(255,255,255),
		float intensity = 10.0f, LIGHT_TYPE type = LIGHT_TYPE::POINT);
	Light(const Light&);
	~Light();
	Light& operator=(const Light&);

	//calculates direction of light from x, y, and z angles
	void calcDirection(float xAngle, float yAngle, float zAngle); 

	//creates an area light shape from a plane
	void createShape(const Plane&);
	
	LIGHT_TYPE type;
	float intensity;
	glm::vec3 color;
	glm::vec3 pos;
	glm::vec3 dir = glm::vec3(0,0,-1);
	bool castsShadow = true;
	bool isAreaLight = false;

	Plane* areaShape;
};

