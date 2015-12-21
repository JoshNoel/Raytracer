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
#include <random>
#include <thread>
#include <mutex>
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

	struct renderThread
	{
	public:
		std::thread m_thread;
		int start;
		int end;
		glm::vec3* data;

		renderThread() {};
		~renderThread()
		{
			if(m_thread.joinable())
				m_thread.join();
		};
	};

	const Scene* scene;
	glm::vec3 castRay(Ray& ray, float& thit0, float& thit1, int depth) const;
	glm::vec3 castRay(Ray& ray, int depth) const;

	bool hitsObject(Ray& ray, float& thit0, float& thit1) const;
	bool hitsObject(Ray& ray) const;

	void startThread(renderThread*) const;
	mutable std::mutex mutex;

	mutable std::mt19937 rng;
	std::uniform_real_distribution<float> distributionX;
	std::uniform_real_distribution<float> distributionY;

	static const float SHADOW_RAY_LENGTH;
	static const int NUM_THREADS;

	mutable std::atomic<int> pixelsRendered;

};

