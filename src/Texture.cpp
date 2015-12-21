#include "Texture.h"
#include <string>
#include <iostream>

Texture::Texture()
	: image(5, 5)
{}

Texture::~Texture()
{}

bool Texture::loadImage(const std::string& path)
{
	std::vector<unsigned char> rawImage;
	//the raw pixels
	unsigned width, height;

	//decode
	unsigned error = lodepng::decode(rawImage, width, height, path);

	//if there's an error, display it
	if(error)
	{
		std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
		return false;
	}

	image.~Image();
	new(&image) Image(width, height);

	//Convert RGBA colors in char array, to RGB in a 2-dimensional array of 3-D integer vectors
	//stores every pixel color linearly by vertical scanline
	std::vector<glm::vec3> imageData;
	for(int i = 0; i < rawImage.size(); i+=4)
	{
		imageData.push_back(glm::vec3(rawImage[i], rawImage[i + 1], rawImage[i + 2]));
	}

	//copy imageData vector to the member image's data pointer
	memcpy(image.data, &imageData[0], sizeof(glm::vec3) * imageData.size());

	return true;
}

const glm::vec3 Texture::getPixel(const glm::vec2& uvCoord) const
{
	//transform normalized uvCoordinates to xy coordinates on image
	int x = int(std::floor(uvCoord.x * image.width));
	int y = int(std::floor(uvCoord.y * image.height));
	glm::vec3 vec =  image.data[y *image.height + x];

	return vec;
}
 