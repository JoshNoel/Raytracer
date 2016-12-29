#include "Shape.h"


Shape::Shape(const glm::vec3& pos)
	: position(pos)
{
	aabb = new BoundingBox();
}

Shape::Shape()
	: position(glm::vec3(0,0,0))
{
	aabb = new BoundingBox();
}


Shape::~Shape()
{
	delete aabb;
}
