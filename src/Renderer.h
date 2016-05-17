#pragma once
#include "Ray.h"
#include "Sphere.h"
#include "Image.h"
#include "Camera.h"
#include "Light.h"
#include "Plane.h"
#include "Scene.h"
#include "ThreadPool.h"
#include <vector>
#include <memory>
#include <random>
#include <atomic>


class Renderer
{
public:
	Renderer(const Scene* scene, Image* image);
	~Renderer();

	void render();

	Image* image;
	Camera camera;

private:

	const Scene* scene;

	//casts ray into the scene and returns color at intersection point if there is an intersection
		//thit0 and thit1 are inputs that are set to the t-values representing the intersection point on the ray
		//depth represents what order ray is being cast (i.e. primary, secondary, tertiary, etc.)
	glm::vec3 castRay(Ray& ray, float& thit0, float& thit1, int depth) const;
	glm::vec3 castRay(Ray& ray, int depth) const;

	//returns whether a ray hits an object
	bool hitsObject(Ray& ray, float& thit0, float& thit1) const;
	bool hitsObject(Ray& ray) const;

	//handles thread management
	ThreadPool threadPool;

	//used to generate random points on an area light to crate soft shadows
	mutable std::minstd_rand rng;
	mutable std::uniform_real_distribution<float> distributionX;
	mutable std::uniform_real_distribution<float> distributionY;

	static const float SHADOW_RAY_LENGTH;
	static const int NUM_THREADS;

	mutable std::atomic<int> pixelsRendered;
};

