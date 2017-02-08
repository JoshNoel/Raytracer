#pragma once
#include "Ray.h"
#include "BoundingBox.h"
#include "managed.h"

//Describes shape that is a part of a GeometryObject
	//abstract class
class Shape
{
public:

	enum SHAPE_TYPE
	{
		TRIANGLE_MESH = 0,
		TRIANGLE,
		SPHERE,
		PLANE,
		NUM_SHAPES
	};

	CUDA_HOST CUDA_DEVICE virtual ~Shape();

	//returns whether a ray intersects the shape
	CUDA_DEVICE virtual bool intersects(Ray& ray, float& thit0, float& thit1) const = 0;

	//returns normal at intersection point
	CUDA_DEVICE virtual glm::vec3 calcWorldIntersectionNormal(const Ray& ray) const = 0;

	//returns shape's type
	CUDA_DEVICE CUDA_HOST virtual SHAPE_TYPE getType() const = 0;

	//position has getters and setters to allow cacheing of triangle coordinates in world space
	CUDA_DEVICE CUDA_HOST virtual glm::vec3 getPosition() const
	{
		return position;
	}

	CUDA_DEVICE CUDA_HOST virtual void setPosition(const glm::vec3& v)
	{
		position = v;
	}

	//Axis-aligned bounding box of the shape
	BoundingBox* aabb;

	//pointer to parent GeometryObj
	//GeometryObj* parent;

	//holds parameters for Shape Class constructors
	//data can only be set through constructors as this sets num_params
		//1 constructor w/o each default parameter for class
	//Only params that are set through constructor are copied to GPU.
	//Struct itself is not copied due to CUDA restrictions on virtual classes,
		//and it saves space as unset data does not need to be copied
	struct parameters
	{
		CUDA_HOST CUDA_DEVICE parameters() {}
		CUDA_HOST CUDA_DEVICE virtual ~parameters(){}


		//returns additive size of constructor parameters for copy to GPU
		//ex: if parameters = (float, int, int), and num_params = 2,
			//then getParamsSize() returns (sizeof(float) + sizeof(int))
		virtual size_t getParamSize() = 0;
		virtual size_t getParamSize(int num_params) = 0;


		CUDA_HOST CUDA_DEVICE int getNumParams() const { return num_params; }
		virtual void* getParam(unsigned i) = 0;

	protected:

		//holds number of parameters actually set within the struct
		//ex: if largest constructor is construct(float, int, int), 
		//but the constructor called is construct(float, int), then num_params=2.
		//MIN <= num_params <= MAX
		//MIN = (# params) - (# default params)
		//MAX = (# params)
		int num_params = 0;
	};

protected:

	CUDA_HOST CUDA_DEVICE Shape(const glm::vec3& pos);
	CUDA_HOST CUDA_DEVICE Shape();

	glm::vec3 position;
};

