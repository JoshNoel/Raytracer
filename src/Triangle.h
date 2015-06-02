#pragma once
#include "glm\glm.hpp"
#include "Shape.h"
#include <array>

class TriObject;

class Triangle : public Shape
{
public:
	Triangle(const std::array<glm::vec3, 3>& p, TriObject* parent);
	~Triangle();

	SHAPE_TYPE getType() const override;
	glm::vec3 calcObjectNormal() const;
	glm::vec3 calcIntersectionNormal(glm::vec3) const override;
	std::array<glm::vec3, 3> getWorldCoords() const;
	glm::vec3 getWorldPos() const;

	bool intersects(Ray& ray, float* thit) const;

	std::array<glm::vec3, 3> points;

private:
	TriObject* parent;
};


