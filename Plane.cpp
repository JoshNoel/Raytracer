#include "Plane.h"


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


Plane::~Plane()
{
}
