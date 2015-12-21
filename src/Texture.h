#pragma once
#include <string>
#include "lodepng.h"
#include "glm\glm.hpp"
#include "Image.h"

class Texture
{
public:
	Texture();
	~Texture();

	//Path should not have extra slashes as escape characters
	//	needs '\' not '\\'
	bool loadImage(const std::string& path);

	//uvCoord stores normalized coordinates
	const glm::vec3 getPixel(const glm::vec2& uvCoord) const;

private:
	Image image;
};