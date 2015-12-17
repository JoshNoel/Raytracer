#pragma once
#include <memory>
#include <vector>
#include "Light.h"
#include "glm\glm.hpp"
#include "GeometryObj.h"
#include "Node.h"
#include "TriObject.h"

class Scene
{
public:
	Scene();
	~Scene();

	void addObject(std::unique_ptr<GeometryObj> &o) 
	{ 
		o->id = idCounter++;
		objectList.push_back(std::move(o));
	}
	void addLight(const Light& l){ lightList.push_back(l); }

	void setAmbient(glm::vec3 color, float intensity)
	{
		ambientColor = color;
		ambientIntensity = intensity;
	}

	void setBgColor(glm::vec3 col)
	{
		bgColor = col;
	}

	glm::vec3 ambientColor = glm::vec3(255,255,255);
	float ambientIntensity = 0.01f;
	glm::vec3 bgColor = glm::vec3(0,0,0);

	std::vector<std::unique_ptr<GeometryObj>> objectList;
	std::vector<Light> lightList;

	int idCounter = 0;

	const int MAX_RECURSION_DEPTH = 1;
	const int SHADOW_SAMPLES = 16;
	const int PRIMARY_SAMPLES = 4;
private:
	BoundingBox sceneBox;
};