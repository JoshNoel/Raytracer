#pragma once
#include "glm/glm.hpp"
#include "Ray.h"
#include "Texture.h"

class Material
{
public:

	//Bit flags allow for combination of material types
	enum MAT_TYPE
	{
		DIFFUSE = 1<<0,
		MIRROR = 1<<1,
		BPHONG_SPECULAR = 1<<2,
		REFRACTIVE = 1<<3,
	};

	//contains realistic indices of refraction for reference
	struct IOR
	{
		static const float AIR;
		static const float WATER;
		static const float ICE;

	private:
		IOR(){}
		~IOR(){}
	};

	//default colors for easy use
	struct COLORS
	{
		static const glm::vec3 WHITE;
	private:
		COLORS() {}
		~COLORS(){}
	};

	//allows for elements of MAT_TYPE enum to act as bit flags
	friend inline MAT_TYPE operator|(MAT_TYPE A, MAT_TYPE B)
	{
		return static_cast<MAT_TYPE>(static_cast<int>(A) | static_cast<int>(B));
	}

	Material(glm::vec3 color = glm::vec3(180, 180, 180), float diffuseCoefficient = 1.0f, glm::vec3 specularColor = COLORS::WHITE,
		float specularCoefficient = 0.8f, float shininess = 1.0f, float reflectivity = 0.2f, float indexOfRefraction = 1.0f);
	~Material();

	MAT_TYPE type = DIFFUSE;

	//calculates unpolarized reflectivity of material from Index of Refraction 1(n1) into the material given the angle of incidence
		// default incoming index of refraction of 1.0f represents air
	float calcReflectivity(float angleOfIncidence, float n1 = 1.0f);

	//returns diffuse color of material at intersection point
		//samples a texture if one exists for the material
		//otherwise returns color
	glm::vec3 sample(const Ray& ray, float t) const;

	//loads .mtl file, which contains material data for corresponding .obj
		//called from GeometryObj::loadOBJ
	bool loadMTL(const std::string& path, const std::string& materialName);

	glm::vec3 color;
	glm::vec3 specularColor;
	float diffuseCoef;
	float specCoef;
	float indexOfRefrac;

	float shininess;
	float reflectivity;

	void setTexture(const Texture&);

private:
	Texture texture;
	bool hasTexture = false;
};
