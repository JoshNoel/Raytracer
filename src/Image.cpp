#include "Image.h"
#include <fstream>
#include "lodepng.h"

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

/*BAD IMPLEMENTATION
//////BMP///////
//	NAME	 ||	BYTES	||	DESC
//---------------------------------------------------
//BMP-Header ||	14		||	Ensures BMP format & size
//DIB-Header ||			||	Detailed Desc
void Image::output(std::string path)
{
BITMAPFILEHEADER bfh;
BITMAPINFOHEADER bih;

//Identifies file as BMP type "BM"
bfh.bfType = 'MB';
//Sets the size of the bmp file. Structs + size of color data
bfh.bfSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + sizeof(data);
//Reserved bits should not be used
bfh.bfReserved1 = 0;
bfh.bfReserved2 = 0;
//bytes from start of file header to bitmap data
bfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);

//sets width and height of bitmap
//height is negative to make index 0,0 correspond to the top-left of the image instead of bottom-left
bih.biHeight = -height;
bih.biWidth = width;
//sets the size of the struct
bih.biSize = sizeof(BITMAPINFOHEADER);
//must be 1
bih.biPlanes = 1;
//Sets bit depth of the bitmap(2^24 possible colors, 256 values per channel)
bih.biBitCount = 24;
//Sets compression of bitmap(uncompressed)
bih.biCompression = BI_RGB;
//Sets size of image in bytes
bih.biSizeImage = sizeof(data);
//Sets default dpi values of 96
bih.biXPelsPerMeter = 0x0ec4;
bih.biYPelsPerMeter = 0x0ec4;

//Specifies that all possible colors in 24 bit depth can be used
bih.biClrUsed = 0;
bih.biClrImportant = 0;


////////IMAGE OUTPUT//////////////
//number of rows = height
//each row padded to multiple of 4 bits
int rowsize = sizeof(glm::vec3)*width;
rowsize += (rowsize % 4);
//pixel array size is rowSize*number of rows(height)
size_t pixelArraySize = rowsize * height;

//creates byte array and initializes values to 0
unsigned char* pixels = new unsigned char[pixelArraySize];
memset(pixels, 0, pixelArraySize);

//stores size of one componenet of glm::vec3, which represents the size of a color channel
size_t channelSize = sizeof(float);
//stores size of a vec3, or unpadded pixel
size_t vecSize = sizeof(glm::vec3);
//iterates over every pixel in data and puts it into the pixel array in bgr order
for(unsigned int i = 0; i < height; ++i)
{
for(unsigned int j = 0; j < width; ++j)
{
//rowsize*i gives byte index of row
//j*vecSize gives byte offset of the start of the pixel in the current row from 0 to width*sizeof(pixelData)
//Adding channel size gives the byte offset to the current color channel
pixels[rowsize * i + j*vecSize] = data[i*width + j].b;
pixels[rowsize * i + j*vecSize] = data[i*width + j].g;
pixels[rowsize * i + j*4] = data[i*width + j].r;
}
}

//creates output stream
std::ofstream ofs;
ofs.open(path.c_str());

//writes the file header, info header, and pixel data
ofs.write((char*)&bfh, sizeof(bfh));
ofs.write((char*)&bih, sizeof(bih));
ofs.write((char*)pixels, pixelArraySize);

ofs.close();
}*/
