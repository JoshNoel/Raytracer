#pragma once
#include "Ray.h"
#include "Sphere.h"
#include "Image.h"
#include <vector>
#include "Camera.h"
#include "Light.h"
#include "Plane.h"
#include <memory>
#include "Scene.h"

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
	const float SHADOW_RAY_LENGTH = 20.0f;
};

