#include "Renderer.h"
#include "Plane.h"
#include "Sphere.h"
#include "GeometryObj.h"
#include "Logger.h"


void main(char** args)
{
	Image image(800, 800);

	Scene scene;
	scene.bgColor = glm::vec3(125, 125, 125);

	//create plane shape (ground)
	Plane planeShape = Plane(glm::vec3(0, -1.8, -12), 0, 0, 0, glm::vec2(13,13));

	//create plane's material
	Material planeMat = Material();
	planeMat.color = glm::vec3(155, 62, 152);
	planeMat.diffuseCoef = 0.2f;
	planeMat.type = Material::DIFFUSE;

	//create plane object that holds shape and material
	std::unique_ptr<GeometryObj> plane = std::make_unique<GeometryObj>(&planeShape, 
		planeMat);
	//scene.addObject(std::move(plane));

	/////SHAPES/////
	
	//create sphere shape
	Sphere sphereShape = Sphere(glm::vec3(-0.6, -0.8, -13), 1.0f);

	//create sphere's material
	Material sphereMat = Material();
	sphereMat.color = glm::vec3(238, 42, 140);
	sphereMat.reflectivity = 0.8f;
	sphereMat.specCoef = 0.8f;
	sphereMat.shininess = 50.0f;
	sphereMat.type = Material::DIFFUSE | Material::BPHONG_SPECULAR;

	//create sphere object that holds shape and material
	std::unique_ptr<GeometryObj> sphere = std::make_unique<GeometryObj>(&sphereShape, 
		sphereMat);
	//scene.addObject(std::move(sphere));


	Sphere sphereShape1 = Sphere(glm::vec3(0, -0.8, -5), 0.5f);
	Material sphereMat1 = Material();
	sphereMat1.color = glm::vec3(0, 200, 0);
	sphereMat1.reflectivity = 0.8f;
	sphereMat1.specCoef = 0.8f;
	sphereMat1.shininess = 100.0f;
	sphereMat1.diffuseCoef = 0.005f;
	sphereMat1.indexOfRefrac = Material::IOR::WATER;
	sphereMat1.type = Material::REFRACTIVE;
	std::unique_ptr<GeometryObj> sphere1 = std::make_unique<GeometryObj>(&sphereShape1,
		sphereMat1);
	//scene.addObject(std::move(sphere1));


	Sphere sphereShape2 = Sphere(glm::vec3(2.1, 1.1, -10), 1.0f);
	Material sphereMat2 = Material();
	sphereMat2.color = glm::vec3(247, 195, 73);
	sphereMat2.specCoef = 0.8f;
	sphereMat2.shininess = 50.0f;
	sphereMat2.type = Material::DIFFUSE | Material::BPHONG_SPECULAR;
	std::unique_ptr<GeometryObj> sphere2 = std::make_unique<GeometryObj>(&sphereShape2,
		sphereMat2);
	//scene.addObject(std::move(sphere2));

	Texture texture;
	texture.loadImage("./docs/textures/Test2.png");
	TriObject* dragon = new TriObject(glm::vec3(0, -.5, -8.5));
	dragon->flipNormals(false);
	Material dragonMat = Material();
	dragonMat.color = glm::vec3(200, 0, 0);
	dragonMat.specCoef = 0.8f;
	dragonMat.shininess = 50.0f;
	dragonMat.type = Material::DIFFUSE | Material::BPHONG_SPECULAR;
	//dragonMat.setTexture(texture);
	
	if(dragon->loadOBJ("./docs/models/bmw.obj"))
	{
		Logger::startClock();
		dragon->initAccelStruct();
		Logger::record("BVH Construction Time");

		scene.addObject(std::make_unique<GeometryObj>(dragon, dragonMat));
	}


	//////LIGHTS//////
	Light light;
	light.type = Light::POINT;
	light.pos = glm::vec3(-1.0f, 8.0f, -2.0f);
	light.color = glm::vec3(255, 197, 143);
	light.intensity = 6.0f;
	Plane lightPlane = Plane(light.pos, degToRad(150.0f), degToRad(-30.0f), 0.0f, glm::vec2(5.0f, 5.0f));
	light.createShape(lightPlane);
	light.isAreaLight = true;
	scene.addLight(light);	

	Light light2;
	light2.type = Light::DIRECTIONAL;
	light2.calcDirection(degToRad(-35.0f), degToRad(-40.0f), 0);
	light2.color = glm::vec3(255, 197, 143);
	light2.intensity = 0.2f;
	//scene.addLight(light2);


	Renderer renderer(&scene, &image);

	scene.setAmbient(glm::vec3(255, 255, 255), 0.01f);


	Logger::startClock();
	renderer.render();
	Logger::record("Render Time");


	image.outputPNG("./docs/examples/test.png");
	Logger::printLog("./docs/logs/Timing_Log_BVH.txt", "Original: Cube");
}