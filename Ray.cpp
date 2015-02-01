#include "Ray.h"

Ray::Ray()
	: pos(0,0,0), dir(0,0,-1)
{
}

Ray::Ray(glm::vec3 p, glm::vec3 d)
	: pos(p), dir(d)
{
}


Ray::~Ray()
{
}
