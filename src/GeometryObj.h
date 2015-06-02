#pragma once
#include "Object.h"
#include "Shape.h"
#include <memory>

class GeometryObj : public Object
{
public:
	GeometryObj(Shape* s, const Material& mat);
	~GeometryObj();

	inline Material& getMaterial() { return material; }
	
	OBJECT_TYPE getType() const override;
	inline Shape* getShape() const { return shape;  }
	bool intersects(Ray& ray, float* thit) const override;


protected:

	Material material;
	Shape* shape;
};

