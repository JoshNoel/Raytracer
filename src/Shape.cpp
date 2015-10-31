#include "Shape.h"


Shape::Shape(const glm::vec3& pos)
	: position(pos), aabb()
{
}

Shape::~Shape()
{
}