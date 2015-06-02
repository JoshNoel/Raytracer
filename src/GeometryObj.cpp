#include "GeometryObj.h"

GeometryObj::GeometryObj(Shape* s, const Material& mat)
	: material(mat), shape(s), Object()
{
}

GeometryObj::~GeometryObj()
{
}

Object::OBJECT_TYPE GeometryObj::getType() const
{
	return OBJECT_TYPE::GEOMETRY;
}

bool GeometryObj::intersects(Ray& ray, float* thit) const
{
	return shape->intersects(ray, thit);
}

