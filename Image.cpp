#include "Image.h"
#include <fstream>
Image::Image(int w, int h)
	: width(w), height(h)
{
	data = new glm::vec3[height * width];
	for(unsigned i = 0; i < width*height; ++i)
	{
		data[i] = glm::vec3(0, 0, 0);
	}
}


Image::~Image()
{
}

void Image::output(std::string path)
{
	std::ofstream ofs;
	ofs.open(path.c_str());
	ofs << "P6\n" << this->width << " " << this->height << "\n255\n";
	for(unsigned i = 0; i < this->height; ++i)
	{
		for(unsigned j = 0; j < this->width; ++j)
		{
			char col[3];
			col[2] = data[i*width+j].r;
			col[0] = data[i*width+j].g;
			col[1] = data[i*width+j].b;
			ofs.write(col, 3);
		}
	}
	ofs.close();
}
