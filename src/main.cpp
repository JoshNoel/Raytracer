#include "Renderer.h"
#include "Plane.h"
#include "Sphere.h"
#include "GeometryObj.h"
#include "Logger.h"
#include <iostream>
#include "Core.h"
#include <stdlib.h>
#include "helper/resource_helper.h"
#include "CudaLoader.h"

int main() {
	//TODO: Determine register limit for render_kernel to balance occupancy and register usage
	Logger::enabled = false;
	Logger::title = "Test Log Entry";

#ifdef USE_CUDA
	//temporary hard-coded values to ensure we don't run out of memory
	//TODO: dynamically determine space needed
	size_t stackSize = 10000; //10kb stack size based on function requirements
	cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
	size_t sSize;
	size_t hSize;
	cudaDeviceGetLimit(&sSize, cudaLimitStackSize);
	cudaDeviceGetLimit(&hSize, cudaLimitMallocHeapSize);
	std::cout << "Stack Size: " << sSize << ", Heap Size: " << hSize << std::endl;
#endif
	//create image and set output path
	Image* image = new Image(1920, 1080);
    std::string outputImagePath = get_image_path("CPU_test.png");

    Camera* camera = new Camera();
	camera->setPosition(glm::vec3(0, 0, 1));

	//create scene and set background color or image
	//Scene.h holds some configuration properties such as samples per pixel
	Scene* scene = new Scene();
	scene->setBgColor(glm::vec3(135, 206, 250));


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

	Plane** planeShape = cudaLoader.loadShape<Plane>(glm::vec3(0, -2.5, -2), 0.0f, 0.0f, 0.0f, glm::vec2(15,20));
	//create plane's material
	Material planeMat;
	planeMat.color = glm::vec3(128, 118, 115);
	planeMat.diffuseCoef = 0.8f;
	planeMat.type = Material::DIFFUSE;

	//create plane object that holds shape and material
	objectList.push_back(std::make_unique<GeometryObj>(planeShape, planeMat, "Plane"));

	Sphere** sphereShape = cudaLoader.loadShape<Sphere>(glm::vec3(-1, -1.5, -7.5), 1);
	Material sphereMat;
	sphereMat.reflectivity = 0.6f;
	sphereMat.type = Material::MIRROR | Material::BPHONG_SPECULAR;
	sphereMat.specCoef = 0.2f;
	sphereMat.shininess = 20.0f;
	sphereMat.specularColor = glm::vec3(255, 255, 255);
	objectList.push_back(std::make_unique<GeometryObj>(sphereShape, sphereMat, "Sphere_mirror"));

	Sphere** sphereShape2 = cudaLoader.loadShape<Sphere>(glm::vec3(1, -1, -6), 1);
	Material sphereMat2;
	sphereMat2.color = glm::vec3(200, 200, 200);
	sphereMat2.diffuseCoef = 0.9f;
	sphereMat2.type = Material::REFRACTIVE | Material::DIFFUSE;
	sphereMat2.specCoef = 0.1f;
	sphereMat2.specularColor = glm::vec3(255, 255, 255);
	sphereMat2.indexOfRefrac = Material::IOR::GLASS;
	objectList.push_back(std::make_unique<GeometryObj>(sphereShape2, sphereMat2, "Sphere_clear"));

	Sphere** sphereShape3 = cudaLoader.loadShape<Sphere>(glm::vec3(-2, 0.1f, -5), 0.5f);
	Material sphereMat3;
	sphereMat3.color = glm::vec3(255, 127, 80);
	sphereMat3.diffuseCoef = 0.8f;
	sphereMat3.type = Material::DIFFUSE | Material::BPHONG_SPECULAR;
	sphereMat3.specCoef = 0.2f;
	sphereMat3.shininess = 25.0f;
	sphereMat3.specularColor = glm::vec3(255, 255, 255);
	objectList.push_back(std::make_unique<GeometryObj>(sphereShape3, sphereMat3, "Sphere_diffuse"));

	//create an area light to illuminate the scene
	//area light is a plane
	//area lights allow for soft shadows because
	//intensity of the shadow depends on area of light that is visible to the point
	glm::vec3 lightpos = glm::vec3(1, 4.0f, 0);
	Plane** lightPlane = cudaLoader.loadShape<Plane>(lightpos, degToRad(-50.0f), degToRad(0.0f), degToRad(0.0f), glm::vec2(5.0f, 5.0f));
	std::unique_ptr<Light> light = std::make_unique<Light>();
	light->type = Light::POINT;
	light->pos = lightpos;
	light->color = glm::vec3(255, 197, 143);
	light->intensity = 20.0f;
	light->setShape(lightPlane);
	light->isAreaLight = true;
	scene->addLight(light.get());

	glm::vec3 lightpos2 = glm::vec3(-1, 3.0f, -9.0);
	Plane** lightPlane2 = cudaLoader.loadShape<Plane>(lightpos2, degToRad(180.0f), degToRad(0.0f), degToRad(0.0f), glm::vec2(10.0f, 10.0f));
	std::unique_ptr<Light> light2 = std::make_unique<Light>();
	light2->type = Light::POINT;
	light2->pos = lightpos2;
	light2->color = glm::vec3(255, 215, 0);
	light2->intensity = 20.0f;
	light2->setShape(lightPlane2);
	light2->isAreaLight = true;
	scene->addLight(light2.get());

	std::unique_ptr<Light> light3 = std::make_unique<Light>();
	light3->type = Light::DIRECTIONAL;
	light3->color = glm::vec3(121, 169, 247);
	light3->intensity = 0.5f;
	light3->calcDirection(-45.0f, 20.0f, 0.0f);
	scene->addLight(light3.get());


	//create list of objects(meshes and materials) from .obj file, and add objects to the scene
	bool flipNormals = false;
	Logger::startClock("Load Time");
	GeometryObj::loadOBJ(cudaLoader, get_obj_path("icosphere.obj"), &objectList, glm::vec3(2.5, -1, -9), flipNormals);
	//GeometryObj::loadOBJ(cudaLoader, get_obj_path("box2.obj"), &objectList, glm::vec3(0, -.7, -7), flipNormals);
	Logger::record("Load Time");

	for (unsigned int i = 0; i < objectList.size(); ++i)
	{
		scene->addObject(objectList[i].get());
	}

	cudaLoader.loadShapePointers();
	light->finalize();
	light2->finalize();
	cudaLoader.loadData(objectList);

	//create renderer with the initialized scene and image pointers
	std::unique_ptr<Renderer> renderer = std::make_unique<Renderer>(scene, image, camera);
	//create core to handle assigning of rendering tasks
	Core core(renderer.get());

	//sets ambient lighting of the scene
		//minimum possible color of an unlit point
	scene->setAmbient(glm::vec3(255, 255, 255), 0.01f);

	//start logger, and then tell core to start rendering
	Logger::startClock("Render Time");
	core.render();
	Logger::record("Render Time");


	std::cout << "Writing image at: " << outputImagePath << std::endl;
	image->outputPNG(outputImagePath);
	Logger::printLog(get_log_path("Timing_Log_cuda.txt"));


	delete image;
	delete camera;
	delete scene;

#ifdef USE_CUDA
	//To fix current issue in NVIDIA Visual Profiler
	CUDA_CHECK_ERROR(cudaDeviceSynchronize());
	CUDA_CHECK_ERROR(cudaProfilerStop());
#endif
	std::cout << "Done" << std::endl;

	return 0;
}



