#include "Object.h"


Object::Object(glm::vec3 pos, Material mat)
	: position(pos), material(mat), aabb(BoundingBox())
{
}

Object::Object()
	: Object(glm::vec3(0, 0, 0), Material())
{
}

Object::~Object()
{
}

