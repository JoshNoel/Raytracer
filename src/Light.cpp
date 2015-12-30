#include "Light.h"
#include "glm\gtx\rotate_vector.hpp"

Light::Light(const glm::vec3& p, const glm::vec3& c, float i, LIGHT_TYPE type)
	: pos(p),
	color(c),
	intensity(i),
	type(type)
{
	areaShape = nullptr;
}

Light::~Light()
{
	delete areaShape;
}

Light::Light(const Light& light)
{
	type = light.type;
	intensity = light.intensity;
	color = light.color;
	pos = light.pos;
	dir = light.dir;
	castsShadow = light.castsShadow;
	isAreaLight = light.isAreaLight;
	if(light.areaShape && light.isAreaLight)
	{
		areaShape = new Plane;
		*areaShape = *light.areaShape;
	}
	else
	{
		areaShape = nullptr;
	}
}

Light& Light::operator=(const Light& light)
{
	type = light.type;
	intensity = light.intensity;
	color = light.color;
	pos = light.pos;
	dir = light.dir;
	castsShadow = light.castsShadow;
	isAreaLight = light.isAreaLight;
	if(light.areaShape && light.isAreaLight)
	{
		areaShape = new Plane;
		*areaShape = *light.areaShape;
	}
	else
	{
		areaShape = nullptr;
	}

	return *this;
}

void Light::calcDirection(float xAngle, float yAngle, float zAngle)
{
	glm::vec3 vector = glm::vec3(0, 0, -1);
	vector = glm::rotateX(vector, xAngle);
	vector = glm::rotateY(vector, yAngle);
	vector = glm::rotateZ(vector, zAngle);

	dir = glm::normalize(vector);
}

void Light::createShape(const Plane& shape)
{
	areaShape = new Plane(shape);
	areaShape->setPosition(pos);
}
