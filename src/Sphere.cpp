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

bool Sphere::intersects(Ray& ray, float& thit0, float& thit1) const
{
	glm::vec3 toRayPos = (ray.pos - this->position);
	float dot = glm::dot(ray.dir, toRayPos);

	//test for intersections
	if(!Math::solveQuadratic(1.0f, 2.0f*dot, glm::dot(toRayPos, toRayPos) - this->radius*this->radius, thit0, thit1))
		return false;
	//if both intersections are behind ray return false
	if(thit0 < 0 && thit1 < 0)
		return false;
	//make t0 the closer point, in front of ray position
	if(thit1 < thit0)
		std::swap(thit1, thit0);
	if(thit0 < 0.0f)
		std::swap(thit0, thit1);
	return true;
}

glm::vec3 Sphere::calcWorldIntersectionNormal(const Ray& ray) const
{
	glm::vec3 intPos = ray.pos + ray.dir * ray.thit0;
	return glm::normalize(intPos - this->position);
}

Sphere::~Sphere()
{
}
