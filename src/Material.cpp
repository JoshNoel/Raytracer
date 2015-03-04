#include "Material.h"


Material::Material(glm::vec3 col, float dc)
	: color(col), diffuseCoef(dc)
{
}


Material::Material()
	: color(161, 161, 161), diffuseCoef()
{
}


Material::~Material()
{
}
