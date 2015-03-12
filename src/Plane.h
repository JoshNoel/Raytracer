#pragma once
#include "Object.h"
/*
*(P-P₀)·n=0
*P = input
*P₀ = point on plane (center in this case)
*n = normal vector of plane
*
*/
class Plane :
	public Object
{
public:
	Plane();
	Plane(glm::vec3, glm::vec3, Material = Material());
	~Plane();

	OBJECT_TYPE getType() const override{ return OBJECT_TYPE::PLANE; }
	bool intersects(const Ray ray, float& t0, float& t1) const override;
	glm::vec3 calcNormal(glm::vec3 p0) const override;

	glm::vec3 normal;
};

