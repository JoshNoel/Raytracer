#include "Plane.h"
#include "MathHelper.h"
#include "glm\glm.hpp"
#include "glm\gtx\rotate_vector.hpp"

Plane::Plane(glm::vec3 position, float xAngle, float yAngle, float zAngle, glm::vec2 dims)
	: Shape(position)
{
	setProperties(xAngle, yAngle, zAngle, dims);
}

void Plane::setProperties(float xAngle, float yAngle, float zAngle, glm::vec2 dims)
{
	//Calculate U,V,N
	glm::vec3 tempNorm(0, 1, 0);
	tempNorm = glm::rotateX(tempNorm, xAngle);
	tempNorm = glm::rotateY(tempNorm, yAngle);
	tempNorm = glm::rotateZ(tempNorm, zAngle);
	this->normal = glm::normalize(tempNorm);

	glm::vec3 tempU = glm::vec3(1, 0, 0);
	tempU = glm::rotateX(tempU, xAngle);
	tempU = glm::rotateY(tempU, yAngle);
	tempU = glm::rotateZ(tempU, zAngle);
	this->uVec = glm::normalize(tempU);

	glm::vec3 tempV = glm::vec3(0, 0, -1);
	tempV = glm::rotateX(tempV, xAngle);
	tempV = glm::rotateY(tempV, yAngle);
	tempV = glm::rotateZ(tempV, zAngle);
	this->vVec = glm::normalize(tempV);

	//seDimensions
	dimensions = glm::vec3(dims.x, PLANE_DEPTH, dims.y);
	aabb.minBounds = position - ((uVec + vVec + normal) * (dimensions / 2.0f));
	aabb.maxBounds = position + ((uVec + vVec + normal) * (dimensions / 2.0f));
	int gh = 0;
}

glm::vec2 Plane::getDimensions() const
{
	return glm::vec2(dimensions.x, dimensions.z);
}

bool Plane::intersects(Ray& ray, float& thit0, float& thit1) const
{
	float val = glm::dot(ray.dir, this->normal);

	//To intersect the plane the ray direction must point opposite to the normal
	if(val < 0)
	{
		//Point on Ray = origin + direction * t
		//point of intersection = origin + direction * thit
		//Plane: 0 = (point - position) • normal
		//	Any line on plane must be at 90 degrees to the plane's normal
		//	therefore dot product between line and normal is 0
		//Substitute point on ray for point in plane equation
		//	0 = (origin + dt - position) • normal
		//	0 = (origin - position) • normal + normal • (direction * t)
		// -normal • direction * t = (origin - position) • normal
		// t = -((origin - position) • normal) / (normal • direction)
		// t = ((position - origin) • normal) / (normal • direction)
		float temp = -glm::dot((ray.pos-position), this->normal) / val;

		//position of intersection using ray equation
		glm::vec3 intersection = ray.pos + (ray.dir * temp);
		if(temp < thit0 &&
			intersection.x > aabb.minBounds.x && intersection.x < aabb.maxBounds.x && 
			intersection.y > aabb.minBounds.y && intersection.y < aabb.maxBounds.y &&
			intersection.z < aabb.minBounds.z && intersection.z > aabb.maxBounds.z)
		{
			thit0 = thit1 = temp;
			return true;
		}
	}
	return false;
}

glm::vec3 Plane::calcWorldIntersectionNormal(glm::vec3) const
{
	return this->normal;
}


Plane::~Plane()
{
}
