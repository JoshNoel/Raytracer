#pragma once
#include "Ray.h"
#include "BoundingBox.h"
#include "managed.h"

class GeometryObj;

//Describes shape that is a part of a GeometryObject
	//abstract class
class Shape : Managed
{
public:

	enum SHAPE_TYPE
	{
		TRIANGLE_MESH = 0,
		TRIANGLE,
		SPHERE,
		PLANE,
		CUBE
	};

	~Shape();

	//returns whether a ray intersects the shape
	virtual bool intersects(Ray& ray, float& thit0, float& thit1) const = 0;

	//returns normal at intersection point
	virtual glm::vec3 calcWorldIntersectionNormal(const Ray& ray) const = 0;

	//returns shape's type
	virtual SHAPE_TYPE getType() const = 0;

	//position has getters and setters to allow cacheing of triangle coordinates in world space
	virtual glm::vec3 getPosition() const
	{
		return position;
	}

	virtual void setPosition(const glm::vec3& v)
	{
		position = v;
	}

	//Axis-aligned bounding box of the shape
	BoundingBox aabb;

	//pointer to parent GeometryObj
	GeometryObj* parent;

protected:

	Shape(const glm::vec3& pos);

	glm::vec3 position;
};

