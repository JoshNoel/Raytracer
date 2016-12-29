#pragma once
#include <string>
#include "CudaDef.h"
#include "glm/glm.hpp"
#include "managed.h"

class Image : public Managed
{
public:
	Image(int w, int h);
	~Image();

	Image(const Image&);
	Image(Image&&);
	Image& operator=(const Image&);
	Image& operator=(Image&&);

	CUDA_HOST CUDA_DEVICE glm::vec3& operator[] (int i) {
		return data[i];
	}

	/*CUDA_HOST CUDA_DEVICE const glm::vec3*& operator[] (int i) const {
		return data[i];
	}*/

	unsigned int width, height;
	long numPixels;

	//returns aspect ration of the image
	inline float getAR() const { return float(width) / height; }

	//outputs data to PPM file at path
	void outputPPM(std::string path) const;

	//outputs data to PNG file at path using lodepng library
	void outputPNG(std::string path) const;

private:
	glm::vec3* data;
};

