#pragma once
#include "Ray.h"
#include "Sphere.h"
#include "Image.h"
#include <vector>
#include "Camera.h"
#include "Light.h"

class Renderer
{
public:
	Renderer();
	Renderer(std::vector<Sphere> spheres, Image* image);
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
	bool testSphere(Ray, Sphere, float& p0, float& p1);
	Image* image;
	Camera camera;

	void addSphere(Sphere s){ sphereList.push_back(s); }
	void addLight(Light l){ lightList.push_back(l); }

private:
	std::vector<Sphere> sphereList;
	std::vector<Light> lightList;

};

