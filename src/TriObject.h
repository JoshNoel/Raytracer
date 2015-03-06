#pragma once
#include "Object.h"
#include <string>
#include <vector>

class TriObject
	: public Object
{
public:
	TriObject(glm::vec3, Material = Material());
	~TriObject();

	bool loadOBJ(std::string path);

	std::vector<glm::vec3> verts;

protected:
	OBJECT_TYPE getType() const override{ return OBJECT_TYPE::TRIANGLE_BASED; };

private:
};