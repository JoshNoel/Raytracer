#include "Sphere.h"
#include "MathHelper.h"

Sphere::Sphere()
	: Sphere(glm::vec3(), 1)
{
}

Sphere::Sphere(glm::vec3 p, float r)
	: radius(r),
	Shape(p)
{
	aabb = BoundingBox(glm::vec3(-radius, -radius, radius) + position,
		glm::vec3(radius, radius, -radius) + position);
}

bool Sphere::intersects(Ray& ray, float* thit0, float* thit1) const
{
	glm::vec3 val = (ray.pos - this->position);
	float t = glm::dot(ray.dir, val);
	//test for intersections
	if(!Math::solveQuadratic(1.0f, 2.0f*glm::dot(ray.dir, val), glm::dot(val, val) - this->radius*this->radius, *thit0, *thit1))
		return false;
	//make x0 the closer point
	if(*thit1 < *thit0)
		std::swap(thit1, thit0);
	return true;
}

glm::vec3 Sphere::calcWorldIntersectionNormal(glm::vec3 intPos) const
{
	return intPos - this->position;
}

Sphere::~Sphere()
{
}
