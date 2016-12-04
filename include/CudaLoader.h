#ifndef CUDA_LOADER_H
#define CUDA_LOADER_H

#include <memory>
#include "Image.h"
#include "Scene.h"
#include "GeometryObj.h"
#include "Material.h"

///Handles transferring data from host to device
class CudaLoader {
public:
	CudaLoader(Scene* scene, Image* image)
		: p_image(image), p_scene(scene)
	{}

	~CudaLoader(){};

	void loadToDevice();

private:
	Image* p_image;
	Scene* p_scene;

	//device pointers
	glm::vec3* pd_image;
	Scene* pd_scene;

	//TODO: Should create structure to hold data in a more efficient manner
};

#endif
