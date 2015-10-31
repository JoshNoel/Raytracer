#pragma once
#include "Material.h"
#include "Shape.h"
#include <memory>


//Combines shape and material into an object
class GeometryObj
{
public:
	GeometryObj(Shape* s, const Material& mat);
	~GeometryObj();

	inline Material& getMaterial() { return material; }
	inline Shape* getShape() const { return shape;  }

	int id = -1;

protected:

	Material material;
	Shape* shape;
};

