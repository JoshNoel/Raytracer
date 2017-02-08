#include "CudaLoader.h"
#include "CudaDef.h"
#include <type_traits>
#include "TriObject.h"
#include "Triangle.h"
#include "Sphere.h"
#include "Plane.h"
#include <functional>
#include "Node.h"
#include <GeometryObj.h>

CUDA_DEVICE void* getAtOffset(void** ptr, size_t offset)
{
	return *reinterpret_cast<void**>(reinterpret_cast<unsigned char*>(ptr) + offset);
}

CUDA_GLOBAL void initKernel(Shape** shapePointerList, Shape::SHAPE_TYPE* shapeTypeList, void** shapeParamList, int* shapeParamLengths, int* numShapes)
{
	//shapePointerList = static_cast<Shape**>(malloc(sizeof(Shape*) * *numShapes));
	
	//holds offset to start of params for indexed shape
	size_t base_offset = 0;
	//for each shape given, create new shape
	for(int i = 0; i < *numShapes; i++)
	{
		int length = shapeParamLengths[i];

		switch (shapeTypeList[i])
		{
		case Shape::SHAPE_TYPE::TRIANGLE_MESH:
			switch (length)
			{
			default:
				printf("TriObject Shape constructor requires 0,1, or 2 parameters!\n");
				break;
			case 0:
				shapePointerList[i] = new TriObject();
				break;
			case 1:
				shapePointerList[i] = new TriObject(*reinterpret_cast<glm::vec3*>(getAtOffset(shapeParamList, base_offset)));
				break;
			case 2:
				shapePointerList[i] = new TriObject(*reinterpret_cast<glm::vec3*>(getAtOffset(shapeParamList, base_offset)), 
					*reinterpret_cast<bool*>(getAtOffset(shapeParamList, base_offset + sizeof(glm::vec3*))));
				break;
			}
			break;
		case Shape::SHAPE_TYPE::PLANE:
			switch (length)
			{
			default:
				printf("Plane Shape constructor requires 0,1,2,3,4, or 5 parameters!\n");
				break;
			case 0:
				shapePointerList[i] = new Plane();
				break;
			case 1:
				shapePointerList[i] = new Plane(*reinterpret_cast<glm::vec3*>(getAtOffset(shapeParamList, base_offset)));
				break;
			case 2:
				shapePointerList[i] = new Plane(*reinterpret_cast<glm::vec3*>(getAtOffset(shapeParamList, base_offset)), 
					*reinterpret_cast<float*>(getAtOffset(shapeParamList, base_offset + sizeof(glm::vec3*))));
				break;
			case 3:
				shapePointerList[i] = new Plane(*reinterpret_cast<glm::vec3*>(getAtOffset(shapeParamList, base_offset)),
					*reinterpret_cast<float*>(getAtOffset(shapeParamList, base_offset + sizeof(glm::vec3*))), 
					*reinterpret_cast<float*>(getAtOffset(shapeParamList, base_offset + sizeof(glm::vec3*) + sizeof(float*))));
				break;
			case 4:
				shapePointerList[i] = new Plane(*reinterpret_cast<glm::vec3*>(getAtOffset(shapeParamList, base_offset)),
					*reinterpret_cast<float*>(getAtOffset(shapeParamList, base_offset + sizeof(glm::vec3*))),
					*reinterpret_cast<float*>(getAtOffset(shapeParamList, base_offset + sizeof(glm::vec3*) + sizeof(float*))),
					*reinterpret_cast<float*>(getAtOffset(shapeParamList, base_offset + sizeof(glm::vec3*) + sizeof(float*)*2)));
				break;
			case 5:
				shapePointerList[i] = new Plane(*reinterpret_cast<glm::vec3*>(getAtOffset(shapeParamList, base_offset)),
					*reinterpret_cast<float*>(getAtOffset(shapeParamList, base_offset + sizeof(glm::vec3*))),
					*reinterpret_cast<float*>(getAtOffset(shapeParamList, base_offset + sizeof(glm::vec3*) + sizeof(float*))),
					*reinterpret_cast<float*>(getAtOffset(shapeParamList, base_offset + sizeof(glm::vec3*) + sizeof(float*) * 2)),
					*reinterpret_cast<glm::vec2*>(getAtOffset(shapeParamList, base_offset + sizeof(glm::vec3*) + sizeof(float*) * 3)));
			}
			break;
		case Shape::SHAPE_TYPE::SPHERE:
			switch (length)
			{
			default:
				printf("Sphere Shape constructor requires 0, 1, or 2 parameters!\n");
				break;
			case 0:
				shapePointerList[i] = new Sphere();
				break;
			case 1:
				shapePointerList[i] = new Sphere(*reinterpret_cast<glm::vec3*>(getAtOffset(shapeParamList, base_offset)));
				break;
			case 2:
				shapePointerList[i] = new Sphere(*reinterpret_cast<glm::vec3*>(getAtOffset(shapeParamList, base_offset)),
					*reinterpret_cast<float*>(getAtOffset(shapeParamList, base_offset + sizeof(glm::vec3*))));
				break;
			}
			break;
		case Shape::SHAPE_TYPE::TRIANGLE:
			switch (length)
			{
			default:
				printf("Triangle Shape constructor requires 2 or 5 parameters!\n");
				break;
			case 2:
				shapePointerList[i] = new Triangle(*reinterpret_cast<helper::array<glm::vec3, 3>*>(getAtOffset(shapeParamList, base_offset)), 
					*reinterpret_cast<bool*>(getAtOffset(shapeParamList, base_offset + sizeof(helper::array<glm::vec3, 3>*))));
				break;
			case 5:
				//helper::array<glm::vec3, 3, false> ar = *reinterpret_cast<helper::array<glm::vec3, 3, false>*>(getAtOffset(shapeParamList, base_offset));
				shapePointerList[i] = new Triangle(*reinterpret_cast<helper::array<glm::vec3, 3, true>*>(getAtOffset(shapeParamList, base_offset)),
					*reinterpret_cast<bool*>(getAtOffset(shapeParamList, base_offset + sizeof(helper::array<glm::vec3, 3>*))),
					*reinterpret_cast<glm::vec3*>(getAtOffset(shapeParamList, base_offset + sizeof(helper::array<glm::vec3, 3>*) + sizeof(bool*))),
					*reinterpret_cast<helper::array<glm::vec2, 3, true>*>(getAtOffset(shapeParamList, base_offset + sizeof(helper::array<glm::vec3, 3>*) + sizeof(bool*) + sizeof(glm::vec3*))), 
					reinterpret_cast<BoundingBox*>(getAtOffset(shapeParamList, base_offset + sizeof(helper::array<glm::vec3, 3>*) + sizeof(bool*) + sizeof(glm::vec3*) + sizeof(helper::array<glm::vec2, 3>*))));
				break;
			}
			break;
		}

		base_offset += shapeParamLengths[i] * sizeof(void*);
	}
}


Triangle** CudaLoader::loadTriangle(Triangle::parameters&& params)
{
#ifdef USE_CUDA
	shapeTypeList.push_back(Shape::SHAPE_TYPE::TRIANGLE);
	shapePointerListSize += sizeof(Triangle);
	shapeParamList.push_back(new Triangle::parameters(std::move(params)));

	trianglePointerList.push_back(nullptr);
	return &trianglePointerList.back();
#else
	return new Triangle(params.points, params.calcNormal, params.position, params.uv_array, params.aabb);
#endif
}

TriObject** CudaLoader::loadTriObject(TriObject::parameters&& params)
{
#ifdef USE_CUDA
	shapeTypeList.push_back(Shape::SHAPE_TYPE::TRIANGLE_MESH);
	shapePointerListSize += sizeof(TriObject);
	shapeParamList.push_back(new TriObject::parameters(std::move(params)));

	triObjectPointerList.push_back(nullptr);
	return &triObjectPointerList.back();
#else
	return new TriObject(params.points, params.calcNormal, params.position, params.uv_array, params.aabb);
#endif
}

Sphere** CudaLoader::loadSphere(Sphere::parameters&& params)
{
#ifdef USE_CUDA
	shapeTypeList.push_back(Shape::SHAPE_TYPE::SPHERE);
	shapePointerListSize += sizeof(Sphere);
	shapeParamList.push_back(new Sphere::parameters(std::move(params)));

	spherePointerList.push_back(nullptr);
	return &spherePointerList.back();
#else
	return new Sphere(params.points, params.calcNormal, params.position, params.uv_array, params.aabb);
#endif
}

Plane** CudaLoader::loadPlane(Plane::parameters&& params)
{
#ifdef USE_CUDA
	shapeTypeList.push_back(Shape::SHAPE_TYPE::PLANE);
	shapePointerListSize += sizeof(Plane);
	shapeParamList.push_back(new Plane::parameters(std::move(params)));

	planePointerList.push_back(nullptr);
	return &planePointerList.back();
	#else
	return new Plane(params.points, params.calcNormal, params.position, params.uv_array, params.aabb);
#endif
}

void CudaLoader::loadShapePointers()
{
#ifdef USE_CUDA
	Shape::SHAPE_TYPE* d_shapeTypeList;
	void** d_shapeParamList;
	int* d_shapeParamLengths;
	int* d_numShapes;
	//size_t* d_shapePointerListSize;
	//size_t* d_shapeParamSizeList;

	std::vector<void*> paramList;
	int totalShapeParamByteSize = std::accumulate(shapeParamSizeList.begin(), shapeParamSizeList.end(), 0);

	//convert vector<Shape::parameters*> -> vector<void*> only holding set data in parameters structs
	for(unsigned i = 0; i < shapeParamList.size(); i++)
	{
		int num_params = shapeParamList[i]->getNumParams();
		shapeParamLengths.push_back(shapeParamList[i]->getNumParams());
		for (int param = 0; param < num_params; param++)
		{
			paramList.push_back(shapeParamList[i]->getParam(param));
		}
	}

	CUDA_CHECK_ERROR(cudaMalloc((void**)&d_shapePointerList, shapeTypeList.size() * sizeof(Shape**)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&d_shapeTypeList, shapeTypeList.size() * sizeof(Shape::SHAPE_TYPE)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&d_shapeParamList, totalShapeParamByteSize));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&d_shapeParamLengths, shapeParamLengths.size() * sizeof(int)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)&d_numShapes, sizeof(int)));
	//CUDA_CHECK_ERROR(cudaMalloc((void**)&d_shapeParamSizeList, sizeof(size_t) * shapeParamSizeList.size()));
	//CUDA_CHECK_ERROR(cudaMalloc((void**)&d_shapePointerListSize, sizeof(size_t)));


	CUDA_CHECK_ERROR(cudaMemcpy(d_shapeTypeList, shapeTypeList.data(), shapeTypeList.size() * sizeof(Shape::SHAPE_TYPE), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(d_shapeParamList, paramList.data(), totalShapeParamByteSize, cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(d_shapeParamLengths, shapeParamLengths.data(), shapeParamLengths.size() * sizeof(int), cudaMemcpyHostToDevice));
	unsigned int numShapes = shapeTypeList.size();
	CUDA_CHECK_ERROR(cudaMemcpy(d_numShapes, &numShapes, sizeof(unsigned int), cudaMemcpyHostToDevice));
	//CUDA_CHECK_ERROR(cudaMemcpy(d_shapeParamSizeList, shapeParamSizeList.data(), sizeof(size_t)*shapeParamSizeList.size(), cudaMemcpyHostToDevice));
	//CUDA_CHECK_ERROR(cudaMemcpy(d_shapePointerListSize, &shapePointerListSize, sizeof(size_t), cudaMemcpyHostToDevice));



	initKernel KERNEL_ARGS2(1, 1) (d_shapePointerList, d_shapeTypeList, d_shapeParamList, d_shapeParamLengths, d_numShapes);

	CUDA_CHECK_ERROR(cudaDeviceSynchronize());

	//shapePointerList pointers now need to point to the device pointers in d_shapePointerList
		//pointer references returned by loadShape will now point to same device pointers as d_shapePointerList
	std::vector<void*> tempShapePointerList;
	tempShapePointerList.resize(shapeTypeList.size());
	CUDA_CHECK_ERROR(cudaMemcpy(tempShapePointerList.data(), d_shapePointerList, shapeTypeList.size() * sizeof(Shape**), cudaMemcpyDeviceToHost));
	std::array<int, Shape::SHAPE_TYPE::NUM_SHAPES> numShapeTypesSet = { 0 };
	for(auto i = 0; i < shapeTypeList.size(); i++)
	{
		switch(shapeTypeList[i])
		{
		case Shape::TRIANGLE:
			trianglePointerList[numShapeTypesSet[Shape::TRIANGLE]] = static_cast<Triangle*>(tempShapePointerList[i]);
			numShapeTypesSet[Shape::TRIANGLE] += 1;
			break;
		case Shape::TRIANGLE_MESH:
			triObjectPointerList[numShapeTypesSet[Shape::TRIANGLE_MESH]] = static_cast<TriObject*>(tempShapePointerList[i]);
			numShapeTypesSet[Shape::TRIANGLE_MESH] += 1;
			break;
		case Shape::SPHERE:
			spherePointerList[numShapeTypesSet[Shape::SPHERE]] = static_cast<Sphere*>(tempShapePointerList[i]);
			numShapeTypesSet[Shape::SPHERE] += 1;
			break;
		case Shape::PLANE:
			planePointerList[numShapeTypesSet[Shape::PLANE]] = static_cast<Plane*>(tempShapePointerList[i]);
			numShapeTypesSet[Shape::PLANE] += 1;
			break;
		}
	}

	CUDA_CHECK_ERROR(cudaDeviceSynchronize());

	CUDA_CHECK_ERROR(cudaFree(d_shapeTypeList));
	CUDA_CHECK_ERROR(cudaFree(d_numShapes));
	CUDA_CHECK_ERROR(cudaFree(d_shapeParamLengths));
	CUDA_CHECK_ERROR(cudaFree(d_shapeParamList));

	CUDA_CHECK_ERROR(cudaDeviceSynchronize());

	for(auto it = shapeParamList.begin(); it != shapeParamList.end(); ++it)
	{
		delete *it;
	}
	shapeParamList.clear();

	shapePointersLoaded = true;
#endif
}

void CudaLoader::queueData(TriObject** destination, const std::vector<Triangle**>& tris, BoundingBox* aabb)
{
	//at this point Shape device pointers are not initialized, so one must still deal with the pointers to them as they will be replaced through loadShapePointers
	triObjectDataList.emplace_back(destination, tris, aabb);
}

CUDA_GLOBAL void data_loader(TriObject** objects_list, unsigned objects_list_size, TriObject::GpuData* data_list)
{
	for(unsigned i = 0; i < objects_list_size; i++)
	{
		objects_list[i]->gpuData = new TriObject::GpuData(data_list[i]);
		objects_list[i]->initAccelStruct();
	}
}

void CudaLoader::loadData(std::vector<std::unique_ptr<GeometryObj>>& geometryObjs)
{
#ifdef USE_CUDA
	//shape pointers must be valid device pointers in order to copy data to them
	//because they are loaded we can now deal with the finalized device Shape* rather than the host pointers to them
	assert(shapePointersLoaded);

	for(auto& obj : geometryObjs)
	{
		obj->finalize();
	}

	//Load Triangle lists for triangle objects
		//create TriObject::gpuData struct for each triObjectData in list
		//copy gpuData and aabb to device, and set gpuData for each device pointers
		//init accel structure for each triObject
	vector<TriObject*> objectList;
	vector<TriObject::GpuData> gpuDataList;
	for(auto data : triObjectDataList)
	{
		//dereference all the host pointers to create vector of just finalized device pointers
		std::vector<Triangle*> tris;
		for(auto tri_ptr_ptr : data.tris)
		{
			tris.push_back(*tri_ptr_ptr);
		}
		gpuDataList.emplace_back(tris, data.aabb);
		objectList.push_back(*data.obj);
	}

	TriObject** d_objectList;
	TriObject::GpuData* d_gpuDataList;

	CUDA_CHECK_ERROR(cudaMalloc((void**)& d_objectList, objectList.size() * sizeof(TriObject*)));
	CUDA_CHECK_ERROR(cudaMalloc((void**)& d_gpuDataList, gpuDataList.size() * sizeof(TriObject::GpuData)));

	CUDA_CHECK_ERROR(cudaMemcpy(d_objectList, objectList.data(), objectList.size() * sizeof(TriObject*), cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR(cudaMemcpy(d_gpuDataList, gpuDataList.data(), gpuDataList.size() * sizeof(TriObject::GpuData), cudaMemcpyHostToDevice));

	data_loader KERNEL_ARGS2(1, 1)(d_objectList, objectList.size(), d_gpuDataList);

	CUDA_CHECK_ERROR(cudaDeviceSynchronize());



#else
	//if not using cuda we can load data to triangle objects and initialize acceleration structure on host
	for(auto data : triObjectDataList)
	{
		data.obj->tris = data.tris;
		data.obj->aabb = data.aabb;
		data.obj->initAccelStruct();
	}
#endif
}

