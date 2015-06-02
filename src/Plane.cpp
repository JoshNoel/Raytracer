#include "Plane.h"
#include "MathHelper.h"
#include "glm\glm.hpp"

Plane::Plane()
	: normal(glm::vec3(0,1,0)),
	Shape(glm::vec3())
{
}

Plane::Plane(glm::vec3 position, glm::vec3 normal)
	: normal(normal),
	Shape(position)
{
}

bool Plane::intersects(Ray& ray, float* thit) const
{
	float val = glm::dot(ray.dir, this->normal);
	if(val < 0)
	{
		float temp = glm::dot((this->position - ray.pos), this->normal) / val;
		if(temp < *thit)
		{
			*thit = temp;
			return true;
		}
	}
	return false;
}

glm::vec3 Plane::calcIntersectionNormal(glm::vec3) const
{
	return this->normal;
}


Plane::~Plane()
{
}
