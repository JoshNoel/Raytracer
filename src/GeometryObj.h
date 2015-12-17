#pragma once
#include "Material.h"
#include "Shape.h"
#include <memory>


//Combines shape and material into an object
class GeometryObj
{
public:
	GeometryObj(std::shared_ptr<Shape> s,  const Material& mat);
	~GeometryObj();

	inline Material& getMaterial() { return material; }
	inline std::shared_ptr<Shape> getShape() const { return shape;  }

	static bool loadOBJ(const std::string& path, std::vector<std::unique_ptr<GeometryObj>>* objects, const glm::vec3& position);

	int id = -1;

protected:

	Material material;
	std::shared_ptr<Shape> shape;
};

