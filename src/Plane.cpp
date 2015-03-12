#include "Plane.h"
#include "MathHelper.h"
#include "glm\glm.hpp"

Plane::Plane()
	: normal(glm::vec3(0,1,0)),
	Object()
{
}

Plane::Plane(glm::vec3 position, glm::vec3 normal, Material mat)
	: normal(normal),
	Object(position, mat)
{
}

bool Plane::intersects(const Ray ray, float& t0, float& t1) const
{
	float val = glm::dot(ray.dir, this->normal);
	if(val < 0)
	{
		t0 = t1 = glm::dot((this->position - ray.pos), this->normal) / val;
		return (t0 > 0);
	}
	return false;
}

glm::vec3 Plane::calcNormal(glm::vec3 p0) const
{
	return this->normal;
}


Plane::~Plane()
{
}
