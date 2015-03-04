#include "Object.h"


Object::Object(glm::vec3 pos, Material mat)
	: position(pos), material(mat)
{
}

Object::Object()
	: position(0, 0, 0), material()
{
}

Object::~Object()
{
}
