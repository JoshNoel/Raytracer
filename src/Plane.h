#pragma once
#include "Shape.h"
#include "MathHelper.h"
/*
*(P-P₀)·n=0
*P = input
*P₀ = point on plane (center in this case)
*n = normal vector of plane
*
*/
class Plane :
	public Shape
{
public:
	Plane();
	Plane(glm::vec3, glm::vec3, glm::vec3 min = glm::vec3(-_INFINITY, -_INFINITY, _INFINITY),
		glm::vec3 max = glm::vec3(_INFINITY, _INFINITY, -_INFINITY));
	~Plane();

	SHAPE_TYPE getType() const override{ return SHAPE_TYPE::PLANE; }
	bool intersects(Ray& ray, float* thit0, float* thit1) const override;
	glm::vec3 calcWorldIntersectionNormal(glm::vec3) const override;

	glm::vec3 normal;
	glm::vec3 minBounds;
	glm::vec3 maxBounds;
};

