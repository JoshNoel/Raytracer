#include "Image.h"
#include <fstream>
#include "lodepng.h"
#include "CudaDef.h"

Image::Image(int w, int h)
	: width(w), height(h)
{
	numPixels = w * h;
#ifdef USE_CUDA
	CUDA_CHECK_ERROR(cudaMallocManaged((void**)&data, sizeof(glm::vec3)*width*height));
#else
	data = new glm::vec3[height * width];
#endif
	for(unsigned i = 0; i < width*height; ++i)
	{
		data[i] = glm::vec3(0, 0, 0);
	}
}


Image::~Image()
{
#ifdef USE_CUDA
	//CUDA_CHECK_ERROR(cudaFree(data));
#else
	delete data;
#endif
}

Image::Image(const Image& image)
	: width(image.width), height(image.height), numPixels(image.numPixels)
{
#ifdef USE_CUDA
	CUDA_CHECK_ERROR(cudaMallocManaged((void**)&data, sizeof(glm::vec3)*width*height));
#else
	data = new glm::vec3[height * width];
#endif
	if (image.data)
		memcpy(&data, &image.data, sizeof(glm::vec3)*width*height);

}


Image::Image(Image&& image)
	: width(image.width), height(image.height), numPixels(image.numPixels)
{
	data = image.data;
	image.data = nullptr;
}

Image& Image::operator=(const Image& image)
{
	if (image.data)
		memcpy(&data, &image.data, sizeof(glm::vec3)*width*height);
	return *this;
}

Image& Image::operator=(Image&& image)
{
	data = image.data;
	image.data = nullptr;
	return *this;
}



void Image::outputPNG(std::string path) const
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

void Image::outputPPM(std::string path) const
{
	std::ofstream ofs;
	ofs.open(path.c_str());
	ofs << "P6\n" << this->width << " " << this->height << "\n255\n";
	for(unsigned i = 0; i < this->height; ++i)
	{
		for(unsigned j = 0; j < this->width; ++j)
		{
			char col[3];
			col[2] = (&data[i])[j].r;
			col[0] = (&data[i])[j].g;
			col[1] = (&data[i])[j].b;
			ofs.write(col, 3);
		}
	}
	ofs.close();
}
