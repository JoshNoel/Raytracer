#include "Sphere.h"


Sphere::Sphere()
	: Object(),
	radius(1)
{
}

Sphere::Sphere(glm::vec3 p, float r, Material mat)
	: radius(r),
	Object(p, mat)
{
}

Sphere::~Sphere()
{
}
