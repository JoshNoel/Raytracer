#pragma once
#include "Material.h"
#include "Shape.h"
#include <memory>


//Combines shape and material into an object
class GeometryObj
{
public:
	GeometryObj(std::shared_ptr<Shape> s, const Material& mat);
	GeometryObj(std::shared_ptr<Shape> s, const Material& mat, const std::string& name);
	~GeometryObj();

	//returns material of the object
	inline Material& getMaterial() { return material; }

	//returns the shape associated with the object
	inline std::shared_ptr<Shape> getShape() const { return shape;  }

	//loads objects and materials  into the objects vector from .obj file at path
	static bool loadOBJ(const std::string& path, std::vector<std::unique_ptr<GeometryObj>>* objects, const glm::vec3& position, bool flipNormals);

	//id set when added to scene
	int id = -1;

	//If imported through GeometryObj::loadOBJ, name contains the name of the object from the .obj file
	std::string name;

protected:

	Material material;
	std::shared_ptr<Shape> shape;
};

