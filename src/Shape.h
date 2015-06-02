#pragma once
#include "Ray.h"
#include "BoundingBox.h"

class Shape
{
public:
	glm::vec3 position;

	enum SHAPE_TYPE
	{
		TRIANGLE_MESH = 0,
		TRIANGLE,
		SPHERE,
		PLANE
	};

	~Shape();

	virtual glm::vec3 calcIntersectionNormal(glm::vec3) const = 0;
	virtual bool intersects(Ray& ray, float* thit) const = 0;
	virtual SHAPE_TYPE getType() const = 0;

	BoundingBox aabb;
protected:
	Shape(glm::vec3 pos);

};

