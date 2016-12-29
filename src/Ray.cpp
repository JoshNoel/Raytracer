#include "Ray.h"
#include "MathHelper.h"

CUDA_HOST CUDA_DEVICE Ray::Ray()
	: pos(0,0,0), thit0(_INFINITY), thit1(-_INFINITY), hitTri(false)
{
	hitObject = nullptr;
	dir = pos + glm::vec3(0, 0, -1);
}

CUDA_HOST CUDA_DEVICE Ray::Ray(glm::vec3 p, glm::vec3 d)
	: pos(p), dir(d), thit0(_INFINITY), thit1(-_INFINITY), hitTri(false)
{
	hitObject = nullptr;
}


CUDA_HOST CUDA_DEVICE Ray::~Ray()
{
}
