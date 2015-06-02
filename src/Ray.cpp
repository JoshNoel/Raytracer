#include "Ray.h"
#include "MathHelper.h"

Ray::Ray()
	: pos(0,0,0), thit(-INFINITY)
{
	dir = pos + glm::vec3(0, 0, -1);
}

Ray::Ray(glm::vec3 p, glm::vec3 d)
	: pos(p), dir(d), thit(-INFINITY)
{
}


Ray::~Ray()
{
}
