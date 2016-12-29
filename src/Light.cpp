#include "Light.h"
#include "CudaDef.h"
#include "glm/gtx/rotate_vector.hpp"

Light::Light(Plane** areaShape, const glm::vec3& p, const glm::vec3& c, float i, LIGHT_TYPE type)
	: pos(p),
	color(c),
	intensity(i),
	type(type),
	host_areaShape(areaShape),
	areaShape(nullptr)
{}

Light::Light(Plane** areaShape)
	: Light(areaShape, glm::vec3(0,0,0))
{}

Light::~Light()
{
}

Light::Light(const Light& light)
	: areaShape(light.areaShape), host_areaShape(light.host_areaShape), type(light.type), intensity(light.intensity), color(light.color), pos(light.color), dir(light.dir),
	castsShadow(light.castsShadow), isAreaLight(light.isAreaLight)
{}

CUDA_DEVICE CUDA_HOST void Light::calcDirection(float xAngle, float yAngle, float zAngle)
{
	glm::vec3 vector = glm::vec3(0, 0, -1);
	vector = glm::rotateX(vector, xAngle);
	vector = glm::rotateY(vector, yAngle);
	vector = glm::rotateZ(vector, zAngle);

	dir = glm::normalize(vector);
}
