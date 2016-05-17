#include "Renderer.h"
#include "Plane.h"
#include "Sphere.h"
#include "GeometryObj.h"
#include "Logger.h"
#include <iostream>
#include <stdlib.h>

///In current setup it will render code_example.png///
int main()
{
	//create image and set output path
	Image image(800, 800);
        std::string outputImagePath = "./docs/examples/test.png";

	//create scene and set background color or image
	Scene scene;
	scene.bgColor = glm::vec3(10, 10, 10);

	//create plane shape (ground)
	std::shared_ptr<Plane> planeShape = std::make_shared<Plane>(glm::vec3(0, -1.5, 0), 0, 0, 0, glm::vec2(150,150));

	//create plane's material
	Material planeMat = Material();
	planeMat.color = glm::vec3(145, 39, 53);
	planeMat.diffuseCoef = 0.3f;
	planeMat.type = Material::DIFFUSE;

	//create plane object that holds shape and material
	std::unique_ptr<GeometryObj> plane = std::make_unique<GeometryObj>(planeShape, planeMat);
	scene.addObject(std::move(plane));
	
	//create list of objects(meshes and materials) from .obj file, and add objects to the scene
	std::vector<std::unique_ptr<GeometryObj>> objectList;
	bool flipNormals = false;
	if(GeometryObj::loadOBJ("./docs/models/cone.obj", &objectList, glm::vec3(0, 0, -10), flipNormals))
	{
		for(int i = 0; i < objectList.size(); ++i)
		{
			scene.addObject(std::move(objectList[i]));
		}
	}


	//create an area light to illuminate the scene
		//area light is a plane
		//area lights allow for soft shadows because 
			//intensity of the shadow depends on area of light that is visible to the point
	Light light;
	light.type = Light::POINT;
	light.pos = glm::vec3(0, 0, 0);
	light.color = glm::vec3(255, 197, 143);
	light.intensity = 10.0f;
	Plane lightPlane = Plane(light.pos, degToRad(-120.0f), degToRad(0.0f), 0.0f, glm::vec2(15.0f, 15.0f));
	light.createShape(lightPlane);
	light.isAreaLight = true;
	scene.addLight(light);	

	//create renderer with the initialized scene and image pointers
	Renderer renderer(&scene, &image);

	//sets ambient lighting of the scene
		//minimum possible color of an unlit point
	scene.setAmbient(glm::vec3(255, 255, 255), 0.1f);


	Logger::startClock();
	renderer.render();
	Logger::record("Render Time");


	image.outputPNG(outputImagePath);
	Logger::printLog("./docs/logs/Timing_Log_example.txt", "Example");


	return 0;
}
