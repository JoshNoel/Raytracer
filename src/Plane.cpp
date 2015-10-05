#include "Plane.h"
#include "MathHelper.h"
#include "glm\glm.hpp"

Plane::Plane()
	: normal(glm::vec3(0,1,0)),
	Shape(glm::vec3())
{
}

Plane::Plane(glm::vec3 position, glm::vec3 normal, glm::vec3 min, glm::vec3 max)
	: normal(normal),
	Shape(position),
	minBounds(min),
	maxBounds(max)
{
	aabb.minBounds = minBounds;
	aabb.maxBounds = maxBounds;
}

bool Plane::intersects(Ray& ray, float* thit0, float* thit1) const
{
	float val = glm::dot(ray.dir, this->normal);

	//To intersect the plane the ray direction must point opposite to the normal
	if(val < 0)
	{
		//Point on Ray = origin + direction * t
		//point of intersection = origin + direction * thit
		//Plane: 0 = (point - position) • normal
		//	Any line on plane must be at 90 degrees to the plane's normal
		//	therefore dot product between line and normal is 0
		//Substitute point on ray for point in plane equation
		//	0 = (origin + dt - position) • normal
		//	0 = (origin - position) • normal + normal • (direction * t)
		// -normal • direction * t = (origin - position) • normal
		// t = -((origin - position) • normal) / (normal • direction)
		// t = ((position - origin) • normal) / (normal • direction)
		float temp = glm::dot((this->position - ray.pos), this->normal) / val;

		//position of intersection using ray equation
		glm::vec3 intersection = ray.pos + (ray.dir * temp);
		if(temp < *thit0 &&
			intersection.x > minBounds.x && intersection.x < maxBounds.x && 
			intersection.y > minBounds.y && intersection.y < maxBounds.y &&
			intersection.z < minBounds.z && intersection.z > maxBounds.z)
		{
			*thit0 = *thit1 = temp;
			return true;
		}
	}
	return false;
}

glm::vec3 Plane::calcWorldIntersectionNormal(glm::vec3) const
{
	return this->normal;
}


Plane::~Plane()
{
}
