#include "Renderer.h"
#include "Plane.h"
#include "Sphere.h"
#include "GeometryObj.h"
#include "Logger.h"


void main(char** args)
{
	Image image(800, 800);

	Scene scene;


	/*Plane planeShape = Plane(glm::vec3(0, -2, -10), glm::vec3(0, 1, 0));
	std::unique_ptr<GeometryObj> plane = std::make_unique<GeometryObj>(&planeShape, Material(glm::vec3(255, 0, 0)));
	scene.addObject(std::move(plane));

	Sphere sphereShape = Sphere(glm::vec3(0, 0, -6), 1);
	std::unique_ptr<GeometryObj> sphere = std::make_unique<GeometryObj>(&sphereShape, Material(glm::vec3(57, 166, 213)));
	scene.addObject(std::move(sphere));
	
	Sphere sphere2Shape = Sphere(glm::vec3(.9, 0.5, -8), 1);
	std::unique_ptr<GeometryObj> sphere2 = std::make_unique<GeometryObj>(&sphere2Shape, Material(glm::vec3(200, 0, 0)));
	scene.addObject(std::move(sphere2));*/


	//SCENE.CREATEKDTREE


	TriObject* dragon = new TriObject(glm::vec3(0, -1, -6));
	if(dragon->loadOBJ("./docs/models/dragon.obj"))
	{
		dragon->initAccelStruct();
		scene.addObject(std::make_unique<GeometryObj>(dragon, Material(glm::vec3(200, 0, 0))));
	}

	Renderer renderer(&scene, &image);



	/*for(unsigned i = 0; i < 5; ++i)
	{
		std::unique_ptr<Object> o = std::make_unique<Sphere>(
			glm::vec3(rand() % 8 - 4, rand() % 8 - 4, -(rand() % 12 + 4)),
			1,
			glm::vec3(rand() % 255, rand() % 255, rand() % 255));
		renderer.addObject(std::move(o));
	}*/

	Light light;
	light.pos = glm::vec3(-1.f, 4.0f, -3.0f);
	light.color = glm::vec3(255, 197, 143);
	light.intensity = 1.0f;
	scene.addLight(light);	

	scene.setAmbient(glm::vec3(255, 255, 255), 0.1f);

	renderer.render();

	/*Logger::startClock();
	Logger::elapsed("Render Time");

	Logger::startClock();*/



	image.outputPNG("./docs/examples/ReflectionOutput.png");
	/*Logger::elapsed("Image Output Time");

	Logger::printLog("./docs/logs/Timing_Log.txt", "Reflection");*/
}