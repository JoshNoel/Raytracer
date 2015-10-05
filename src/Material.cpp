#include "Material.h"


Material::Material(glm::vec3 col, float ior, float dc)
	: color(col), diffuseCoef(dc), indexOfRefrac(ior)
{
}


Material::Material()
	: color(161, 161, 161), diffuseCoef(), indexOfRefrac(1.0f)
{
}


Material::~Material()
{
}

float Material::calcReflectivity(float angle, float n1)
{
	float angleOfRefraction = std::asinf((n1*std::sin(angle)) / this->indexOfRefrac);
	
	float i1cos = n1*std::cosf(angle);
	float r2cos = this->indexOfRefrac * std::cosf(angleOfRefraction);
	float Rs = std::powf(std::fabsf((i1cos - r2cos) / (i1cos + r2cos)), 2.0f);
	
	float i2cos = n1*std::cosf(angleOfRefraction);
	float r1cos = this->indexOfRefrac * std::cosf(angle);
	float Rp = std::powf(std::fabsf((i2cos - r1cos) / (i2cos + r1cos)), 2.0f);

	return (Rs + Rp) / 2.0f;
}
