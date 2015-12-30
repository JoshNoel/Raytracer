#pragma once
#include <string>
#include "lodepng.h"
#include "glm\glm.hpp"
#include "Image.h"

//Holds an image that can be added to a material and uv-mapped to a mesh
class Texture
{
public:
	Texture();
	~Texture();

	//Path should not have extra slashes as escape characters
	//	needs '\' (or '/') not '\\' (or '//')
	bool loadImage(const std::string& path);

	//uvCoord stores normalized coordinates
	const glm::vec3 getPixel(const glm::vec2& uvCoord) const;

private:
	Image image;
};