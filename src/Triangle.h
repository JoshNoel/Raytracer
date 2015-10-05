#pragma once
#include "glm\glm.hpp"
#include <array>
#include "Shape.h"

class TriObject;

//Holds 3 points that create triangle
class Triangle : public Shape
{
public:
	Triangle(const std::array<glm::vec3, 3>& p);
	~Triangle();
	SHAPE_TYPE getType() const override { return SHAPE_TYPE::TRIANGLE; }


	glm::vec3 calcObjectNormal() const;
	glm::vec3 calcWorldIntersectionNormal(glm::vec3) const override;
	std::array<glm::vec3, 3> getWorldCoords() const;
	bool intersects(Ray& ray, float* thit0, float* thit1) const override;
	glm::vec3 getWorldPos() const;

	std::array<glm::vec3, 3> points;
};


