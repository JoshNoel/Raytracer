#pragma once
#include "Shape.h"
#include "MathHelper.h"

#ifndef PLANE_DEPTH
	#define PLANE_DEPTH 0.01f
#endif

/*
*(P-P₀)·n=0
*P = input
*P₀ = point on plane (center in this case)
*n = normal vector of plane
*
*/
class Plane :
	public Shape
{
public:
	//position: position of plane
	//xAngle, yAngle, zAngle: angles to rotate normal vector by in radians
	//	default normal is (0,1,0)
	Plane(glm::vec3 position = glm::vec3(0,0,0), float xAngle = 0.0f, float yAngle = 0.0f, float zAngle = 0.0f, glm::vec2 dimensions = glm::vec2(1000.0f, 1000.0));
	~Plane();

	SHAPE_TYPE getType() const override{ return SHAPE_TYPE::PLANE; }
	bool intersects(Ray& ray, float& thit0, float& thit1) const override;
	glm::vec3 calcWorldIntersectionNormal(const Ray& ray) const override;

	void setProperties(float xAngle, float yAngle, float zAngle, glm::vec2 dimensions);
	glm::vec2 getDimensions() const;
	glm::vec3 getNormal() const { return normal; }
	glm::vec3 getU() const { return uVec; }
	glm::vec3 getV() const { return vVec; }

private:
	//x = width
	//	distance along uVec
	//y = length
	//	distance along vVec
	glm::vec2 dimensions;
	glm::vec3 normal;

	//represents normalized "x-axis", of plane's coordinate system
	glm::vec3 uVec;
	//represents normalized "y-axis", of plane's coordinate system
	glm::vec3 vVec;
};

