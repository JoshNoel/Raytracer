#pragma once
#include "Shape.h"
#include <string>
#include <vector>
#include "Triangle.h"
#include "Node.h"
#include "CudaDef.h"
#include "managed.h"

class CudaLoader;

class TriObject
	: public Shape
{
public:

	CUDA_DEVICE CUDA_HOST TriObject(glm::vec3 pos = glm::vec3(0,0,0), bool flipNormals = false);

	struct parameters : public Shape::parameters
	{

		parameters(const glm::vec3& position, bool flip_normals)
		{
			data = new Data();
			num_params = 2; 
			data->position = position;
			data->flipNormals = flip_normals;
		}

		explicit parameters(const glm::vec3& position)
		{
			num_params = 1;
			data = new Data();
			data->position = position;
		}

		parameters()
		{
			num_params = 0;
			data = new Data();
		}

		parameters(parameters&& params)
		{
			data = new Data();
			data->position = params.getPosition();
			data->flipNormals = params.getFlipNormals();
			num_params = params.num_params;
		}

		glm::vec3 getPosition() const { return data->position;  }
		bool getFlipNormals() const { return data->flipNormals;  }

		void* getParam(unsigned i) override 
		{
			switch (i)
			{
			default:
				assert(i < num_params);
			case 0:
				return &data->position;
			case 1:
				return &data->flipNormals;
			}
		}

		size_t getParamSize(int num_params) override
		{
			assert(num_params < MAX_PARAMS);
			return PARAM_SIZES[num_params];
		}

		size_t getParamSize() override
		{
			return PARAM_SIZES[num_params];
		}

	private:
		static const int MAX_PARAMS = 3;
		//holds additive size of params
		static const int PARAM_SIZES[MAX_PARAMS];
		struct Data : public Managed {
			glm::vec3 position = glm::vec3(0, 0, 0);
			bool flipNormals = false;
		};
		Data* data;
	};

	//TODO: Implement TriObject Copy Constructor
	CUDA_DEVICE CUDA_HOST TriObject(const TriObject&) = delete;
	CUDA_DEVICE CUDA_HOST ~TriObject();

	static bool loadOBJ(std::vector<Triangle**>& tris, BoundingBox* aabb, glm::vec3 pos, const std::string& path, CudaLoader& loader);
	//startLine: start of object data in .obj file
	//vertexNum: set to number to offset vertice index by in 'f' lines because obj does not reset vertex indices by object
	//	on return set to number of vertices in object in order to update offset for next objects
	//uvNum: set to number to offset uv index by in 'f' lines because obj does not reset uv indices by object
	//	on return set to number of uv coordinates in object in order to update offset for next objects
	//if using CUDA, tris is filled with a vector of uninitialized device pointers
    static bool loadOBJ(std::vector<Triangle**>& tris, BoundingBox* aabb, glm::vec3 pos, std::string path, int startLine, std::string& materialName, int& vertexNum, int& uvNum, CudaLoader& loader);

	//initializes acceleration structure from triangle array
	CUDA_DEVICE void initAccelStruct();

#ifndef USE_CUDA
	vector<Triangle**> tris;
#endif
	Node* root;
	
	//tests for intersection using BVH tree
	CUDA_DEVICE bool intersects(Ray& ray, float& thit0, float& thit1) const override;
	CUDA_DEVICE glm::vec3 calcWorldIntersectionNormal(const Ray& ray) const override;

	void flipNormals(bool flip)
	{
		this->invertNormals = flip;
	}

#ifdef USE_CUDA
	struct GpuData {
		CUDA_DEVICE CUDA_HOST GpuData(Triangle** tris, size_t trisSize, BoundingBox* bbox)
			: tris(tris), trisSize(trisSize), bbox(bbox)
		{}

		CUDA_DEVICE CUDA_HOST GpuData(const std::vector<Triangle*>& tris, BoundingBox* bbox)
			: bbox(bbox)
		{
			trisSize = tris.size();
			CUDA_CHECK_ERROR(cudaMalloc((void**)&this->tris, sizeof(Triangle*) * trisSize));
			CUDA_CHECK_ERROR(cudaMemcpy(this->tris, tris.data(), sizeof(Triangle*) * trisSize, cudaMemcpyHostToDevice));
		}

		CUDA_DEVICE CUDA_HOST GpuData(GpuData&& g)
		{
			tris = g.tris;
			trisSize = g.trisSize;
			g.tris = nullptr;
			bbox = g.bbox;
			g.bbox = nullptr;
		}

		CUDA_DEVICE CUDA_HOST GpuData(const GpuData& g)
		{
#if (defined(__CUDA_ARCH__)) && (__CUDA_ARCH__ > 0)
			tris = static_cast<Triangle**>(malloc(sizeof(Triangle*) * g.trisSize));
			for(auto i = 0; i < g.trisSize; i++)
			{
				tris[i] = g.tris[i];
			}
#else
			CUDA_CHECK_ERROR(cudaMalloc((void**)&tris, sizeof(Triangle*) * g.trisSize));
			CUDA_CHECK_ERROR(cudaMemcpy(this->tris, g.tris, sizeof(Triangle*) * g.trisSize, cudaMemcpyDeviceToDevice));
#endif
			trisSize = g.trisSize;
			bbox = new BoundingBox(*g.bbox);
		}

		CUDA_DEVICE CUDA_HOST GpuData()
		{
			tris = nullptr;
			trisSize = 0;
			bbox = new BoundingBox();
		}

		CUDA_DEVICE CUDA_HOST ~GpuData()
		{
			CUDA_CHECK_ERROR(cudaFree(tris));
		}
		//vector of device Triangle pointers
		Triangle** tris;
		unsigned trisSize;
		BoundingBox* bbox;
	};
	GpuData* gpuData;
#endif


protected:
	CUDA_DEVICE CUDA_HOST SHAPE_TYPE getType() const override{ return SHAPE_TYPE::TRIANGLE_MESH; };


private:

	CUDA_DEVICE bool checkTris(Node* node, Ray& ray, float& thit0, float& thit1) const;
	CUDA_DEVICE bool checkNode(Node* node, Ray& ray, float& thit0, float& thit1) const;

    char* tempMatName;

	bool invertNormals;


};
