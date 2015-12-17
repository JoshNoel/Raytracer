#include "Material.h"
#include "Ray.h"
#include "Triangle.h"
#include "MathHelper.h"
#include <array>
#include <fstream>

Material::Material(glm::vec3 col, float dc, glm::vec3 specCol, float sc, float shine, float ref, float ior)
	: color(col), diffuseCoef(dc), indexOfRefrac(ior), specCoef(sc),
	specularColor(specCol), shininess(shine), reflectivity(ref), texture()
{
}


Material::~Material()
{
}

bool Material::loadMTL(const std::string& path, const std::string& materialName)
{
	int extStart = path.find_last_of('.');
	if(path.substr(extStart) != ".mtl")
		return false;

	std::string line;

	std::ifstream ifs;
	ifs.open(path);
	if(!ifs.is_open())
		return false;

	//clear file stream until it reaches the correct material name
	//	indicates start of material data
	while(getline(ifs, line))
	{
		if(line.find("newmtl") != std::string::npos && line.find(materialName) != std::string::npos)
			break;
	}

	while(getline(ifs, line))
	{
		if(line.find("map_Kd") != std::string::npos)
		{
			int spacePos = line.find_first_of(' ');
			std::string texturePath = line.substr(spacePos + 1);
			texture.loadImage(texturePath);
		}
		else if(line.find("Ns") != std::string::npos)
		{
			int spacePos = line.find_first_of(' ');
			shininess = std::stof(line.substr(spacePos + 1));
		}
		else if(line.find("Ni") != std::string::npos)
		{
			int spacePos = line.find_first_of(' ');
			indexOfRefrac = std::stof(line.substr(spacePos + 1));
		}
		else if(line.find("Kd") != std::string::npos)
		{
			//convert from normalized colors to [0, 255] scale
			int spacePos = line.find_first_of(' ');
			float r = std::stof(line.substr(spacePos + 1)) * 255.0f;

			spacePos = line.find_first_of(' ');
			float g = std::stof(line.substr(spacePos + 1)) * 255.0f;

			spacePos = line.find_first_of(' ');
			float b = std::stof(line.substr(spacePos + 1)) * 255.0f;

			color = glm::vec3(r, g, b);
		}
		else if(line.find("Ks") != std::string::npos)
		{
			int spacePos = line.find_first_of(' ');
			float r = std::stof(line.substr(spacePos + 1)) * 255.0f;

			spacePos = line.find_first_of(' ');
			float g = std::stof(line.substr(spacePos + 1)) * 255.0f;

			spacePos = line.find_first_of(' ');
			float b = std::stof(line.substr(spacePos + 1)) * 255.0f;

			specularColor = glm::vec3(r, g, b);
		}
	}

	if(ifs.bad())
		return false;

	ifs.close();
	return true;
}


glm::vec3 Material::sample(const Ray& ray, float t) const
{
	if(!hasTexture)
		return color;

	//sample texture
	//first compute coefficients for weighted average of uvCoords at 3 triangle points for the intersection point
	//I = intersection
	glm::vec3 I = ray.pos + ray.dir * t;
	std::array<glm::vec2, 3> triCoords;
	if(!ray.hitTri->getUV(triCoords))
		return color;
	
	/*					    p3
							/\
						   /  \
						  /	   \
						 /	    \
						/	 	 \
					   /		  \
					  /		•	   \
					 /	    	    \
					/			     \
				   /__________________\
				  p1				  p2
	
	
	*/

	//if lines are drawn from each vertex to the intersection point,
	//	the ratio of the uvCoordinate of each point contributed to the intersection 
	//	is equal to the ratio of the are of the opposite inner triangle to the area of the whole triangle
	glm::vec3 p1 = ray.hitTri->getWorldCoords()[0];
	glm::vec3 p2 = ray.hitTri->getWorldCoords()[1];
	glm::vec3 p3 = ray.hitTri->getWorldCoords()[2];

	//areas are doubled, but the ratio of the sub-triangle areas to the whole triangle is all that matters
	glm::vec3 totalAreaVec = glm::cross(p2 - p1, p3 - p1);
	float totalArea = glm::length(totalAreaVec);
	glm::vec3 ItoP1 = p1 - I;
	glm::vec3 ItoP2 = p2 - I;
	glm::vec3 ItoP3 = p3 - I;
	
	float p1Area = glm::length(glm::cross(ItoP2, ItoP3));
	float p2Area = glm::length(glm::cross(ItoP3, ItoP1));
	float p3Area = glm::length(glm::cross(ItoP1, ItoP2));

	float p1Weight = p1Area / totalArea;
	float p2Weight = p2Area / totalArea;
	float p3Weight = p3Area / totalArea;

	glm::vec2 uv1 = triCoords[0];
	glm::vec2 uv2 = triCoords[1];
	glm::vec2 uv3 = triCoords[2];

	glm::vec2 intersectionCoord = uv1 * p1Weight + uv2 * p2Weight + uv3 * p3Weight;
	return texture.getPixel(intersectionCoord);
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

void Material::setTexture(const Texture& tex)
{
	this->texture = tex;
	hasTexture = true;
}

//Initialize constant indicies of refraction
const float Material::IOR::AIR = 1.0f;
const float Material::IOR::WATER = 4.0f/3.0f;
const float Material::IOR::ICE = 1.31f;

//Initialize constant colors
const glm::vec3 Material::COLORS::WHITE = glm::vec3(255, 255, 255);