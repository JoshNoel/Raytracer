#ifndef CUDA_LOADER_H
#define CUDA_LOADER_H

#include "helper/array.h"
#include <vector>
#include <array>
#include <type_traits>
#include "Shape.h"
#include <numeric>
#include "Triangle.h"
#include "TriObject.h"
#include "Sphere.h"
#include "Plane.h"
#include "BoundingBox.h"

///Handles transferring data from host to device
class CudaLoader {
public:

	~CudaLoader()
	{
#ifdef USE_CUDA
		CUDA_CHECK_ERROR(cudaFree(d_shapePointerList));

		for(auto p : shapeParamList)
		{
			delete p;
		}
#endif
	}

	//need to preallocate storage for Shape* vectors, as returned pointers to indices are invalid if the vector has to reallocate its memory due to push_back
	//better to overestimate than underestimate value
	void setNumShapesHint(Shape::SHAPE_TYPE type, int num)
	{
		switch (type)
		{
		case Shape::TRIANGLE:
			trianglePointerList.reserve(sizeof(Triangle*)*num);
			break;
		case Shape::TRIANGLE_MESH:
			triObjectPointerList.reserve(sizeof(TriObject*)*num);
			break;
		case Shape::PLANE:
			planePointerList.reserve(sizeof(Plane*)*num);
			break;
		case Shape::SPHERE:
			spherePointerList.reserve(sizeof(Sphere*)*num);
			break;
		}
	}

	///adds shape to shape pointer list to be allocated on device through loadShapePointers()
	template<class T, class... ARGS>
	typename std::enable_if<std::is_same<T, Triangle>::value, Triangle**>::type
	loadShape(ARGS&&... my_args)
	{
		shapeParamSizeList.push_back(sizeof...(my_args) * sizeof(void*));
		return loadTriangle(Triangle::parameters(std::forward<ARGS>(my_args)...));
	}

	template<class T, class... ARGS>
	typename std::enable_if<std::is_same<T, TriObject>::value, TriObject**>::type 
	loadShape(ARGS&&... my_args)
	{
		shapeParamSizeList.push_back(sizeof...(my_args) * sizeof(void*));
		return loadTriObject(TriObject::parameters(std::forward<ARGS>(my_args)...));
	}

	template<class T, class... ARGS>
	typename std::enable_if<std::is_same<T, Plane>::value, Plane**>::type
	loadShape(ARGS&&... my_args)
	{
		shapeParamSizeList.push_back(sizeof...(my_args) * sizeof(void*));
		return loadPlane(Plane::parameters(std::forward<ARGS>(my_args)...));
	}

	template<class T, class... ARGS>
	typename std::enable_if<std::is_same<T, Sphere>::value, Sphere**>::type
	loadShape(ARGS&&... my_args)
	{
		shapeParamSizeList.push_back(sizeof...(my_args) * sizeof(void*));
		return loadSphere(Sphere::parameters(std::forward<ARGS>(my_args)...));
	}


	Triangle** loadTriangle(Triangle::parameters&&);
	TriObject** loadTriObject(TriObject::parameters&&);
	Sphere** loadSphere(Sphere::parameters&&);
	Plane** loadPlane(Plane::parameters&&);

	void loadShapePointers();


	///Queues data for copy to destination through CudaLoader::loadData()
	void queueData(TriObject** destination, const std::vector<Triangle**>& tris, BoundingBox* aabb);

	///Loads queued data to device
	///Once data is loaded it will initialize TriObject acceleration structures (on host if not using CUDA, in kernel if using)
	///Also finalizes GeometryObj's past to it (GeometryObj::finalize)
	void loadData(std::vector<std::unique_ptr<GeometryObj>>&);

private:
	//holds pointers to address of Shape*'s that are returned by loadShape<T>
		//this way the pointers loaded on device can overwrite the pointers returned by loadShape<T>
		//host can then use loadData to load Shape data post-creation through the device pointers
		//need vectors for each Shape type as we need to return class specific references to pointers in vectors
	std::vector<Triangle*> trianglePointerList;
	std::vector<TriObject*> triObjectPointerList;
	std::vector<Plane*> planePointerList;
	std::vector<Sphere*> spherePointerList;
	//device list of Shape* that will hold the device allocated Shape*'s for the lifetime of CudaLoader
	Shape** d_shapePointerList;

	std::vector<Shape::SHAPE_TYPE> shapeTypeList;
	std::vector<Shape::parameters*> shapeParamList;
	std::vector<int> shapeParamLengths;
	//holds byte size of parameter per shape object
	std::vector<size_t> shapeParamSizeList;
	//holds total size of Shape array that is allocated on device
	size_t shapePointerListSize = 0;
	bool shapePointersLoaded = false;

	//data structures for device copy
	struct TriObjectData
	{
		TriObjectData(TriObject** obj, const std::vector<Triangle**>& tris, BoundingBox* aabb)
			: obj(obj), aabb(aabb)
		{
			this->tris = tris;
		}

		TriObject** obj;
		//vector of host pointer to device pointers of Triangles
		std::vector<Triangle**> tris;
		BoundingBox* aabb;
	};
	//need vectors of data to copy to device
	std::vector<TriObjectData> triObjectDataList;

	//useful functions for loadShape()
	size_t getSize() { return 0;  }
	template<typename HEAD, typename... TAIL>
	size_t getSize(const HEAD& head, const TAIL&... tail)
	{
		return sizeof head + getSize(tail...);
	}

	//source: http://aherrmann.github.io/programming/2016/02/28/unpacking-tuples-in-cpp14/
	template<class T, class Tuple, size_t... Indices>
	T createParams(Tuple tuple, std::index_sequence<Indices...>)
	{
		return T{ std::get<Indices>(tuple)... };
	}
};

#endif
