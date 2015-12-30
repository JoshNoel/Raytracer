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

	//adds an object to the scene and sets its id
	void addObject(std::unique_ptr<GeometryObj> &o) 
	{ 
		o->id = idCounter++;
		objectList.push_back(std::move(o));
	}

	//adds a light to the scene
	void addLight(const Light& l){ lightList.push_back(l); }

	//sets ambient color of the scene
		//minimum color for a point in shadow
	void setAmbient(glm::vec3 color, float intensity)
	{
		ambientColor = color;
		ambientIntensity = intensity;
	}

	//sets background color of the scene
	void setBgColor(glm::vec3 col)
	{
		bgColor = col;
	}

	glm::vec3 ambientColor = glm::vec3(255,255,255);
	float ambientIntensity = 0.01f;
	glm::vec3 bgColor = glm::vec3(0,0,0);

	//store objects and lights that are a part of the scene
	std::vector<std::unique_ptr<GeometryObj>> objectList;
	std::vector<Light> lightList;

	int idCounter = 0;

	const int MAX_RECURSION_DEPTH = 2;

	//samples to cast to an area light to determine shadow intensity
		//if a ray doesn't intersect an object on the way to the light, the point is lit
		//if it does intersect it is in shadow
		//average the results of the samples to determine visibililty of point to an area light
			//1 = completly visible
			//0 = completly in shadow
	const int SHADOW_SAMPLES = 16;

	//samples to cast per pixel
		//also uses stratified random sampling (like with the area lights) within each pixel
		//average color of the primary rays to get final color of the pixel
	const int PRIMARY_SAMPLES = 4;
private:

	//Axis-aligned bounding box for the scene as a whole
	BoundingBox sceneBox;
};