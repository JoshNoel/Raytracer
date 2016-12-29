#include "Renderer.h"
#include "Plane.h"
#include "Sphere.h"
#include "GeometryObj.h"
#include "Logger.h"
#include <iostream>
#include "Core.h"
#include <stdlib.h>
#include "helper/array.h"
#include "CudaLoader.h"

///In current setup it will render code_example.png///
int main() {
	//temporary hard-coded values to ensure we don't run out of memory
	//TODO: dynamically determine space needed
	size_t stackSize = 2e4; //20kb stack size
	size_t heapSize = 4e6; //4gb heap size
	cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
	size_t sSize;
	size_t hSize;
	cudaDeviceGetLimit(&sSize, cudaLimitStackSize);
	cudaDeviceGetLimit(&hSize, cudaLimitMallocHeapSize);
	std::cout << "Stack Size: " << sSize << ", Heap Size: " << hSize << std::endl;

	//create image and set output path
	Image* image = new Image(800, 800);
    std::string outputImagePath = "F:\\Projects\\cuda\\raytracer\\docs\\examples\\CUDA_test.png";

    Camera* camera = new Camera();

	//create scene and set background color or image
	Scene* scene = new Scene();
	scene->setBgColor(glm::vec3(10, 10, 10));


	//need specialized loading functionality if using cuda
	CudaLoader cudaLoader;	

	//cudaLoader needs to know number of Shape*'s that will be loaded to return pointers to the inside of the vector
		//transition to dynamically determined values once scene definition is loaded from file (Can then count shapes defined in file)
		//can also then dynamically determine heap and stack size needed on device
	cudaLoader.setNumShapesHint(Shape::TRIANGLE, 200);
	cudaLoader.setNumShapesHint(Shape::PLANE, 2);
	cudaLoader.setNumShapesHint(Shape::SPHERE, 1);
	cudaLoader.setNumShapesHint(Shape::TRIANGLE_MESH, 1);



	std::vector<std::unique_ptr<GeometryObj>> objectList;


	Plane** planeShape = cudaLoader.loadShape<Plane>(glm::vec3(0, -2, -2), 0.0f, 0.0f, 0.0f, glm::vec2(100,100));

	//create plane's material
	Material planeMat;
	planeMat.color = glm::vec3(128, 118, 115);
	planeMat.diffuseCoef = 0.8f;
	planeMat.type = Material::DIFFUSE;

	//create plane object that holds shape and material
	objectList.push_back(std::make_unique<GeometryObj>(planeShape, planeMat, "Plane"));

	Sphere** sphereShape = cudaLoader.loadShape<Sphere>(glm::vec3(-1, -1, -6), 1);
	Material sphereMat;
	sphereMat.color = glm::vec3(15, 175, 200);
	sphereMat.diffuseCoef = 0.8f;
	sphereMat.type = Material::DIFFUSE | Material::BPHONG_SPECULAR;
	sphereMat.specCoef = 0.2f;
	sphereMat.specularColor = glm::vec3(255, 255, 255);
	objectList.push_back(std::make_unique<GeometryObj>(sphereShape, sphereMat, "Sphere"));

	//create an area light to illuminate the scene
	//area light is a plane
	//area lights allow for soft shadows because
	//intensity of the shadow depends on area of light that is visible to the point
	glm::vec3 lightpos = glm::vec3(0, 5.0f, 0);
	Plane** lightPlane = cudaLoader.loadShape<Plane>(lightpos, degToRad(180.0f), degToRad(0.0f), 0.0f, glm::vec2(15.0f, 15.0f));
	std::unique_ptr<Light> light = std::make_unique<Light>();
	light->type = Light::POINT;
	//light->calcDirection(-45.0f, 0.0f, 0.0f);
	light->pos = lightpos;
	light->color = glm::vec3(255, 197, 143);
	light->intensity = 100.0f;
	light->setShape(lightPlane);
	light->isAreaLight = true;
	scene->addLight(light.get());


	//create list of objects(meshes and materials) from .obj file, and add objects to the scene
	bool flipNormals = false;
	GeometryObj::loadOBJ(cudaLoader, "F:\\Projects\\cuda\\raytracer\\docs\\models\\icosphere.obj", &objectList, glm::vec3(1, -1, -6), flipNormals);

	for (unsigned int i = 0; i < objectList.size(); ++i)
	{
		scene->addObject(objectList[i].get());
	}

	cudaLoader.loadShapePointers();
	light->finalize();
	cudaLoader.loadData(objectList);

	//create renderer with the initialized scene and image pointers
	std::unique_ptr<Renderer> renderer = std::make_unique<Renderer>(scene, image, camera);
	//create core to handle assigning of rendering tasks
	Core core(renderer.get());

	//sets ambient lighting of the scene
		//minimum possible color of an unlit point
	scene->setAmbient(glm::vec3(255, 255, 255), 0.1f);

	//start logger, and then tell core to start rendering
	Logger::startClock();
	core.render();
	Logger::record("Render Time");


	image->outputPNG(outputImagePath);
	Logger::printLog(".\\docs\\logs\\Timing_Log_example.txt", "Example");

	delete image;
	delete camera;
	delete scene;
	return 0;
}



