#pragma once
#include <string>
#include <vector>
#include "glm/glm.hpp"
#include "CudaDef.h"
#include "managed.h"

class Image : Managed
{
public:
	Image(int w, int h);
	~Image();

	int width, height;
	long numPixels;
	glm::vec3* data;

	//returns aspect ration of the image
	inline float getAR(){ return float(width) / height; }

	//outputs data to PPM file at path
	void outputPPM(std::string path);

	//outputs data to PNG file at path using lodepng library
	void outputPNG(std::string path);	
};

