﻿#pragma once
#include "glm\glm.hpp"
#include <array>
#include "Shape.h"

class TriObject;

//Holds 3 points that create triangle
/*************PLANE INTERSECTION*****************
*For any point in a plane P, where P₀ is position of plane, and N is plane's normal:
*Eq.1: (P-P₀) • N = 0
*	This is true because if point is on plane a vector from the plane to this point is
*	orthogonal with the plane's normal
*Eq.2 substitute ray equation into Eq.1: (o + dt - P₀) • N = 0
*Eq.3 distribute dot product in Eq.2: (dt • N) + ((o-P₀) • N) = 0
*E1.4 solve Eq.2 for t: -[((o-P₀) • N)]/[d • N] = t
**************************************************
*************TRIANGLE INTERSECTION****************
*If intersection with plane is inside triangle then,
*the angle between any edge, and the vector between the start of that edge and the intersection
*point is less than 90 degees, so the the dot product between these vectors is positive
*/
class Triangle : public Shape
{
public:
	Triangle(const std::array<glm::vec3, 3>& p);
	~Triangle();
	SHAPE_TYPE getType() const override { return SHAPE_TYPE::TRIANGLE; }


	glm::vec3 calcObjectNormal() const;
	glm::vec3 calcWorldIntersectionNormal(glm::vec3) const override;
	std::array<glm::vec3, 3> getWorldCoords() const;
	bool intersects(Ray& ray, float& thit0, float& thit1) const override;
	glm::vec3 getWorldPos() const;

	std::array<glm::vec3, 3> points;
};

