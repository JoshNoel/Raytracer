#include "Renderer.h"
#include "TriObject.h"
#include "Logger.h"

void main(char** args)
{
	Image image(800, 800);

	Scene scene;


	std::unique_ptr<Object> plane = std::make_unique<Plane>(glm::vec3(0, -2, -10), glm::vec3(0, 1, 0), Material(glm::vec3(255, 0, 0)));
	scene.addObject(std::move(plane));

	std::unique_ptr<Object> sphere = std::make_unique<Sphere>(glm::vec3(0, 0, -6), 1, Material(glm::vec3(57, 166, 213)));
	scene.addObject(std::move(sphere));
	std::unique_ptr<Object> sphere2 = std::make_unique<Sphere>(glm::vec3(.9, 0.5, -8), 1, glm::vec3(200, 0, 0));
	scene.addObject(std::move(sphere2));

	Renderer renderer(&scene, &image);


	/*std::unique_ptr<TriObject> box = std::make_unique<TriObject>(glm::vec3(-1, 1, -6), Material(glm::vec3(222, 157,83)));
	if(!box->loadOBJ("./docs/models/box.obj")){}
	renderer.addObject(std::move(box));*/
	
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

	//scene.initAccelStruct();

	Logger::startClock();
	renderer.render();
	Logger::elapsed("Render Time");

	Logger::startClock();
	image.outputPNG("./docs/examples/Output.png");
	Logger::elapsed("Image Output Time");

	Logger::printLog("./docs/logs/Timing_Log.txt", "No Accel");
}