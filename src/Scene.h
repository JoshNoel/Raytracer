#pragma once
#include <memory>
#include <vector>
#include "Object.h"
#include "Light.h"
#include "glm\glm.hpp"

class Scene
{
public:
	Scene();
	~Scene();

	void addObject(std::unique_ptr<Object> o) { objectList.push_back(std::move(o)); }
	void addLight(Light l){ lightList.push_back(l); }
	//void initAccelStruct();

	void setAmbient(glm::vec3 color, float intensity)
	{
		ambientColor = color;
		ambientIntensity = intensity;
	}

	glm::vec3 ambientColor;
	float ambientIntensity;

	std::vector<std::unique_ptr<Object>> objectList;
	std::vector<Light> lightList;

private:

};