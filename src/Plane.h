#pragma once
#include "Shape.h"
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
	Plane(glm::vec3, glm::vec3);
	~Plane();

	SHAPE_TYPE getType() const override{ return SHAPE_TYPE::PLANE; }
	bool intersects(Ray& ray, float* thit) const override;
	glm::vec3 calcIntersectionNormal(glm::vec3) const;

	glm::vec3 normal;
};

