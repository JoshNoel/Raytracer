#pragma once
#include "Material.h"
#include <memory>
#include "managed.h"
#include "CudaDef.h"
#include "CudaLoader.h"


//Combines shape and material into an object
//Type of Shape* held needs to be known at compile time as it is stored as a reference to pointer
class GeometryObj : public Managed
{
public:
	template<typename T_Shape>
	GeometryObj(T_Shape** s, const Material& mat)
		: material(mat)
	{
		static_assert(std::is_base_of<Shape, T_Shape>::value, "Reference to pointer passed to Constructor must be derived from Shape");
#ifdef USE_CUDA
		p_host_shape = reinterpret_cast<Shape**>(s);
#else
		shape = *reinterpret_cast<Shape**>(s);
#endif
	}

	template<typename T_Shape>
	GeometryObj(T_Shape** s, const Material& mat, const std::string& name)
		: GeometryObj(s, mat)
	{
		this->name = name;
	}

	~GeometryObj();

	//returns material of the object
	CUDA_HOST CUDA_DEVICE inline Material& getMaterial() { return material; }

	//returns the shape associated with the object
	CUDA_HOST CUDA_DEVICE inline Shape* getShape() const { return shape;  }

	//loads objects and materials  into the objects vector from .obj file at path
	static bool loadOBJ(CudaLoader& cudaLoader, const std::string& path, std::vector<std::unique_ptr<GeometryObj>>* objectList, const glm::vec3& position, bool flipNormals);

	//id set when added to scene
	int id = -1;

	//If imported through GeometryObj::loadOBJ, name contains the name of the object from the .obj file
	std::string name;

public:
	void finalize()
	{
		shape = *p_host_shape;
	}

private:
	static const std::string DEFAULT_MTL_PATH;
	static const std::string DEFAULT_MTL_NAME;

protected:
	
	Material material;

	//finalized device pointer set after CudaLoader::loadShapePointers
	Shape* shape;

	//holds temporary host pointer to device pointer until device pointer is set in CudaLoader::loadShapePointers
	Shape** p_host_shape;
};
