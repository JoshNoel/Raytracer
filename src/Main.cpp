#include "Renderer.h"
#include "Plane.h"
#include "Sphere.h"
#include "GeometryObj.h"
#include "Logger.h"
#include "Cube.h"


void main(char** args)
{
	Image image(800, 800);

	Scene scene;
	scene.bgColor = glm::vec3(125, 125, 125);

	//create plane shape (ground)
	Plane planeShape = Plane(glm::vec3(0, -1.0, -10), 0, 0, 0, glm::vec2(50,50));

	//create plane's material
	Material planeMat = Material();
	planeMat.color = glm::vec3(155, 62, 152);
	planeMat.specularColor = Material::COLORS::WHITE;
	planeMat.type = Material::DIFFUSE;

	//create plane object that holds shape and material
	std::unique_ptr<GeometryObj> plane = std::make_unique<GeometryObj>(&planeShape, 
		planeMat);
	scene.addObject(std::move(plane));

	Cube cubeShape = Cube(glm::vec3(0, 0, -6), glm::vec3(1, 1, -1));
	Material cubeMat = Material();
	cubeMat.color = glm::vec3(155, 62, 152);
	cubeMat.specCoef = 0.0f;
	cubeMat.type = Material::DIFFUSE;
	std::unique_ptr<GeometryObj> cube = std::make_unique<GeometryObj>(&cubeShape,
		cubeMat);
	//scene.addObject(std::move(cube));

	//create sphere shape
	Sphere sphereShape = Sphere(glm::vec3(0, 0.5, -6), 0.8f);

	//create sphere's material
	Material sphereMat = Material();
	sphereMat.color = glm::vec3(200, 0, 0);
	sphereMat.specCoef = 0.8f;
	sphereMat.shininess = 100.0f;
	sphereMat.indexOfRefrac = 1.2f;
	sphereMat.type = Material::DIFFUSE;

	//create sphere object that holds shape and material
	std::unique_ptr<GeometryObj> sphere = std::make_unique<GeometryObj>(&sphereShape, 
		sphereMat);
	scene.addObject(std::move(sphere));



	Sphere sphereShape1 = Sphere(glm::vec3(1.5, 0.5, -14), 1);

	Material sphereMat1 = Material();
	sphereMat1.color = glm::vec3(0, 200, 0);
	sphereMat1.specCoef = 0.8f;
	sphereMat1.shininess = 100.0f;
	sphereMat1.type = Material::DIFFUSE | Material::BPHONG_SPECULAR;

	std::unique_ptr<GeometryObj> sphere1 = std::make_unique<GeometryObj>(&sphereShape1,
		sphereMat1);
	//scene.addObject(std::move(sphere1));

	Sphere sphereShape2 = Sphere(glm::vec3(-1.5, 0, -9), 1);
	Material sphereMat2 = Material();
	sphereMat2.color = glm::vec3(0, 0, 200);
	sphereMat2.specCoef = 0.8f;
	sphereMat2.shininess = 100.0f;
	sphereMat2.type = Material::DIFFUSE | Material::BPHONG_SPECULAR;
	std::unique_ptr<GeometryObj> sphere2 = std::make_unique<GeometryObj>(&sphereShape2,
		sphereMat2);
	//scene.addObject(std::move(sphere2));


	/*TriObject* dragon = new TriObject(glm::vec3(0, 0.0f, -5));
	Material dragonMat = Material(glm::vec3(200, 0, 0), 1.0f, 0.1f, 100.0f);
	dragonMat.type = Material::DIFFUSE | Material::BPHONG_SPECULAR;
	if(dragon->loadOBJ("./docs/models/monkey.obj"))
	{
		dragon->initAccelStruct();
		scene.addObject(std::make_unique<GeometryObj>(dragon, dragonMat));
	}*/

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
	light.pos = glm::vec3(-4.0f, 5.0f, 0.0f);
	light.color = glm::vec3(255, 197, 143);
	light.intensity = 10.0f;
	Plane lightPlane = Plane(light.pos, degToRad(130.0f), degToRad(0.0f), 0.0f, glm::vec2(10.0f, 10.0f));
	light.createShape(lightPlane);
	light.isAreaLight = true;
	scene.addLight(light);	

	Light light2;
	light2.type = Light::DIRECTIONAL;
	light2.calcDirection(degToRad(-35.0f), degToRad(-40.0f), 0);
	light2.color = glm::vec3(255, 197, 143);
	light2.intensity = 0.2f;
	//scene.addLight(light2);

	scene.setAmbient(glm::vec3(255, 255, 255), 0.01f);


	Logger::startClock();
	renderer.render();

	Logger::record("Render Time");

	Logger::startClock();



	image.outputPNG("./docs/examples/AreaOutput.png");
	Logger::record("Image Output Time");

	//Logger::printLog("./docs/logs/Timing_Log.txt", "Threads: 10, Copy To Image");
}