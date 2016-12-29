#include "Sphere.h"
#include "MathHelper.h"

const int Sphere::parameters::PARAM_SIZES[Sphere::parameters::MAX_PARAMS] = {
0,
sizeof(glm::vec3),
sizeof(glm::vec3) + sizeof(float) };

Sphere::Sphere(glm::vec3 p, float r)
	: radius(r),
	Shape(p)
{
	aabb = new BoundingBox(glm::vec3(-radius, -radius, radius) + position,
		glm::vec3(radius, radius, -radius) + position);
}

CUDA_DEVICE bool Sphere::intersects(Ray& ray, float& thit0, float& thit1) const
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
	if (thit1 < thit0) {
		auto temp = thit1;
		thit1 = thit0;
		thit0 = temp;
	}

	if (thit0 < 0.0f) {
		auto temp = thit1;
		thit1 = thit0;
		thit0 = temp;
	}
	return true;
}

CUDA_DEVICE glm::vec3 Sphere::calcWorldIntersectionNormal(const Ray& ray) const
{
	glm::vec3 intPos = ray.pos + ray.dir * ray.thit0;
	return glm::normalize(intPos - this->position);
}

Sphere::~Sphere()
{
}
