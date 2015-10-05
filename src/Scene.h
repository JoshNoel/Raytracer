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
		objectList.push_back(std::move(o));
	}
	void addLight(Light l){ lightList.push_back(l); }

	void setAmbient(glm::vec3 color, float intensity)
	{
		ambientColor = color;
		ambientIntensity = intensity;
	}

	glm::vec3 ambientColor;
	float ambientIntensity;

	std::vector<std::unique_ptr<GeometryObj>> objectList;
	std::vector<Light> lightList;
	//std::array<Node>

private:
	BoundingBox sceneBox;
};