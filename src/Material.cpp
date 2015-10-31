#include "Material.h"
#include "Ray.h"

Material::Material(glm::vec3 col, float dc, glm::vec3 specCol, float sc, float shine, float ref, float ior)
	: color(col), diffuseCoef(dc), indexOfRefrac(ior), specCoef(sc),
	specularColor(specCol), shininess(shine), reflectivity(ref)
{
}


Material::~Material()
{
}

glm::vec3 Material::sample(const Ray& ray, float t)
{
	return color;
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

//Initialize constant indicies of refraction
const float Material::IOR::AIR = 1.0f;
const float Material::IOR::WATER = 4.0f/3.0f;
const float Material::IOR::ICE = 1.31f;

//Initialize constant colors
const glm::vec3 Material::COLORS::WHITE = glm::vec3(255, 255, 255);