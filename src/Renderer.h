#pragma once
#include "Ray.h"
#include "Sphere.h"
#include "Image.h"
#include <vector>
#include "Camera.h"
#include "Light.h"
#include "Plane.h"
#include <memory>

class Renderer
{
public:
	Renderer();
	Renderer(std::vector<std::unique_ptr<Object>> objects, Image* image);
	~Renderer();

	void render();

	/*Tests for ray sphere intersection using Math.solveQuadratic(see Sphere.h)
	* returns result of determinant
	*	-1 = no intersection
	*	 0 = ray is tangent
	*	 1 = 2 intersections
	*	
	* p1 will contain position of first intersection(or only if tangent)
	* p2 will contain position of second intersection
	*/
	Image* image;
	Camera camera;

	void addObject(std::unique_ptr<Object> o) { objectList.push_back(std::move(o)); }
	void addLight(Light l){ lightList.push_back(l); }
	
	void setAmbient(glm::vec3 color, float intensity) 
	{ 
		ambientColor = color; 
		ambientIntensity = intensity;
	}
private:
	glm::vec3 ambientColor;
	float ambientIntensity;
	
	std::vector<std::unique_ptr<Object>> objectList;
	std::vector<Light> lightList;

	const float SHADOW_RAY_LENGTH = 20.0f;
};

