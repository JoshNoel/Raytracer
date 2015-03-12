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
	long collTriIndex;

	bool intersects(const Ray ray, float& t0, float& t1) const override;
	glm::vec3 calcNormal(glm::vec3 p0) const override;

protected:
	OBJECT_TYPE getType() const override{ return OBJECT_TYPE::TRIANGLE_BASED; };

private:
};