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

bool Sphere::intersects(Ray& ray, float* thit) const
{
	glm::vec3 val = (ray.pos - this->position);
	float t = glm::dot(ray.dir, val);
	//test for intersections
	float t0, t1;
	if(!Math::solveQuadratic(1.0f, 2.0f*glm::dot(ray.dir, val), glm::dot(val, val) - this->radius*this->radius, t0, t1))
		return false;
	//make x0 the closer point
	t0 < t1 ? *thit = t0 : *thit = t1;
	return true;
}

glm::vec3 Sphere::calcIntersectionNormal(glm::vec3 intPos) const
{
	return intPos - this->position;
}

Sphere::~Sphere()
{
}
