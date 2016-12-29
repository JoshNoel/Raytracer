#pragma once
#include <string>
#include "CudaDef.h"
#include "lodepng.h"
#include "glm/glm.hpp"
#include "Image.h"
#include "managed.h"

//Holds an image that can be added to a material and uv-mapped to a mesh
class Texture : public Managed
{
public:
	Texture();
	~Texture();

	//Path should not have extra slashes as escape characters
	//	needs '\' (or '/') not '\\' (or '//')
	bool loadImage(const std::string& path);

	//uvCoord stores normalized coordinates
	CUDA_HOST CUDA_DEVICE glm::vec3 getPixel(const glm::vec2& uvCoord) const;

private:
	Image* image;
};
