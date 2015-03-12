#include "Sphere.h"
#include "MathHelper.h"

Sphere::Sphere()
	: Sphere(glm::vec3(), 1, Material())
{
}

Sphere::Sphere(glm::vec3 p, float r, Material mat)
	: radius(r),
	Object(p, mat)
{
	aabb.minBounds.x = -radius;
	aabb.minBounds.y = -radius;
	aabb.minBounds.z = radius;

	aabb.maxBounds.x = radius;
	aabb.maxBounds.y = radius;
	aabb.maxBounds.z = -radius;

	aabb.minBounds += position;
	aabb.maxBounds += position;
}

bool Sphere::intersects(const Ray ray, float& t0, float& t1) const
{
	glm::vec3 val = (ray.pos - this->position);
	float t = glm::dot(ray.dir, val);
	//test for intersections
	if(!Math::solveQuadratic(1.0f, 2.0f*glm::dot(ray.dir, val), glm::dot(val, val) - this->radius*this->radius, t0, t1))
		return false;
	//make x0 the closer point
	if(t1<t0) std::swap(t0, t1);
	return true;
}

glm::vec3 Sphere::calcNormal(glm::vec3 p0) const
{
	return p0 - this->position;
}

Sphere::~Sphere()
{
}
