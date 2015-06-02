#pragma once
#include "Shape.h"
#include <string>
#include <vector>
#include "Triangle.h"

class TriObject
	: public Shape
{
public:
	TriObject(glm::vec3);
	~TriObject();

	bool loadOBJ(std::string path);

	std::vector<Triangle> tris;
	mutable int collTriIndex;

	bool intersects(Ray& ray, float* thit) const override;
	glm::vec3 calcIntersectionNormal(glm::vec3) const override;

protected:
	SHAPE_TYPE getType() const override{ return SHAPE_TYPE::TRIANGLE_MESH; };


private:
};