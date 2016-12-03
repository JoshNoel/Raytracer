#pragma once
#include "glm/glm.hpp"
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
*If the ray intersects the intersection point within the triangle formed by the three points
*then cross product between side vector pointing CCW between points, and vector from start of a side
*to intersection will always point in same direction as the normal
*/
class Triangle : public Shape
{
public:
	Triangle(const std::array<glm::vec3, 3>& p, bool calcNormal);
	~Triangle();
	SHAPE_TYPE getType() const override { return SHAPE_TYPE::TRIANGLE; }

	//add uv coordinates to the triangle
	//	coordinate order matches with vertex order in points array
	void setUVCoords(const std::array<glm::vec2, 3>&);

	//returns the precaclulated normal value
	glm::vec3 calcWorldIntersectionNormal(const Ray&) const override;

	//returns points in world space
	std::array<glm::vec3, 3> getWorldCoords() const;

	//returns points in object space
	std::array<glm::vec3, 3> getPoints() const;

	//returns uvCoordinates
	bool getUV(std::array<glm::vec2, 3>& coords) const;

	bool intersects(Ray& ray, float& thit0, float& thit1) const override;

	
	
	glm::vec3 normal;

	bool hasUV = false;

	//updates position, points_world, and triangleCenter
	void setPosition(const glm::vec3& pos) override;

private:
	//used to compute normal at creation
	glm::vec3 calcObjectNormal() const;

	std::array<glm::vec3, 3> points;
	std::array<glm::vec3, 3> points_world;
	std::array<glm::vec2, 3> uvCoords;

	glm::vec3 triangleCenter;

	//calculate center of triangle
	glm::vec3 calcCenter() const;
};


