#pragma once
#include "Ray.h"
#include "BoundingBox.h"

class GeometryObj;

//Describes shape that is a part of a GeometryObject and has a bounding box
class Shape
{
public:
	~Shape();

	BoundingBox aabb;

	enum SHAPE_TYPE
	{
		TRIANGLE_MESH = 0,
		TRIANGLE,
		SPHERE,
		PLANE
	};

	glm::vec3 position;

	virtual bool intersects(Ray& ray, float* thit, float* thit0) const = 0;
	virtual glm::vec3 calcWorldIntersectionNormal(glm::vec3) const = 0;
	virtual SHAPE_TYPE getType() const = 0;

	GeometryObj* parent;

protected:
	Shape(const glm::vec3& pos);

};

