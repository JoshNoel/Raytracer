#include "Sphere.h"


Sphere::Sphere()
	: pos(0,0,-5),
	radius(1)
{
}

Sphere::Sphere(glm::vec3 p, float r, glm::vec3 color)
	: pos(p),
	radius(r),
	color(color)
{
}

Sphere::~Sphere()
{
}
