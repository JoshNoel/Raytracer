#include "Texture.h"
#include <string>
#include <iostream>

Texture::Texture() {
	image = new Image(5, 5);
}

Texture::~Texture() {
	/*
	if(image)
		delete image;
		*/
}

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

	delete image;
	image = new Image(width, height);

	//Convert RGBA colors in char array, to RGB in a 2-dimensional array of 3-D integer vectors
	//stores every pixel color linearly by vertical scanline
	std::vector<glm::vec3> imageData;
	for(unsigned i = 0; i < rawImage.size(); i+=4)
	{
		imageData.push_back(glm::vec3(rawImage[i], rawImage[i + 1], rawImage[i + 2]));
	}

	//copy imageData vector to the member image's data pointer
	memcpy(&(*image)[0], &imageData[0], sizeof(glm::vec3) * imageData.size());

	return true;
}

CUDA_HOST CUDA_DEVICE glm::vec3 Texture::getPixel(const glm::vec2& uvCoord) const
{
	//transform normalized uvCoordinates to xy coordinates on image
	int x = int(std::floor(uvCoord.x * image->width));
	int y = int(std::floor(uvCoord.y * image->height));
	glm::vec3 vec =  (*image)[y *image->height + x];

	return vec;
}
 
