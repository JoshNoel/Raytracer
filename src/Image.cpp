#include "Image.h"
#include <fstream>
#include "lodepng.h"

Image::Image(int w, int h)
	: width(w), height(h)
{
	numPixels = w * h;
	data = new glm::vec3[height * width];
	for(unsigned i = 0; i < width*height; ++i)
	{
		data[i] = glm::vec3(0, 0, 0);
	}
}


Image::~Image()
{
}

void Image::outputPNG(std::string path)
{
	std::vector<unsigned char> pixelData;
	for(unsigned int i = 0; i < width*height; ++i)
	{
		pixelData.push_back(data[i].r);
		pixelData.push_back(data[i].g);
		pixelData.push_back(data[i].b);
		pixelData.push_back(255);
	}
	lodepng::encode(path, pixelData, width, height);
}

void Image::outputPPM(std::string path)
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