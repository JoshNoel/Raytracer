#pragma once
#include <string>
#include <vector>
#include "glm\glm.hpp"
class Image
{
public:
	Image(int w, int h);
	~Image();

	int width, height;
	glm::vec3* data;

	inline float getAR(){ return float(width) / height; }
	void output(std::string path);
	
};

