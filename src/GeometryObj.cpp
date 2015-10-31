#include "GeometryObj.h"

GeometryObj::GeometryObj(Shape* s, const Material& mat)
	: material(mat), shape(s)
{
	s->parent = this;
}

GeometryObj::~GeometryObj()
{
}

