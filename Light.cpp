#include "Light.h"


Light::Light()
	: pos(0, 0, 0),
	color(255, 255, 255),
	intensity(1)
{
}

Light::Light(glm::vec3 p, glm::vec3 c, float i)
	: pos(p),
	color(c),
	intensity(i)
{
}

Light::~Light()
{
}
