#include "CudaDef.h"
#include "Plane.h"
#include "MathHelper.h"
#include "glm/glm.hpp"
#include "glm/gtx/rotate_vector.hpp"

const int Plane::parameters::PARAM_SIZES[Plane::parameters::MAX_PARAMS] = { 
0,
sizeof(glm::vec3),
sizeof(glm::vec3) + sizeof(float),
sizeof(glm::vec3) + sizeof(float) * 2,
sizeof(glm::vec3) + sizeof(float) * 3,
sizeof(glm::vec3) + sizeof(float) * 3 + sizeof(glm::vec2)
};


Plane::Plane(const glm::vec3& position, float xAngle, float yAngle, float zAngle, const glm::vec2& dims)
	: Shape(position)
{
	setProperties(xAngle, yAngle, zAngle, dims);
}

void Plane::setProperties(float xAngle, float yAngle, float zAngle, glm::vec2 dims)
{
	//Calculate U,V,N vectors representing the sides and normal of the triangle
	glm::vec3 tempNorm(0.0f, 1.0f, 0.0f);
	tempNorm = glm::rotateX(tempNorm, xAngle);
	float x = cosf(xAngle);
	glm::vec3 r;
	r.x = tempNorm.x;
	r.y = tempNorm.y * cosf(xAngle) - tempNorm.z * sinf(xAngle);
	r.z = tempNorm.y * sinf(xAngle) + tempNorm.z * cosf(xAngle);
	tempNorm = r;
	tempNorm = glm::rotateY(tempNorm, yAngle);
	tempNorm = glm::rotateZ(tempNorm, zAngle);
	this->normal = glm::normalize(tempNorm);

	glm::vec3 tempU = glm::vec3(1.0f, 0, 0);
	tempU = glm::rotateX(tempU, xAngle);
	tempU = glm::rotateY(tempU, yAngle);
	tempU = glm::rotateZ(tempU, zAngle);
	this->uVec = glm::normalize(tempU);

	glm::vec3 tempV = glm::vec3(0, 0, -1.0f);
	tempV = glm::rotateX(tempV, xAngle);
	tempV = glm::rotateY(tempV, yAngle);
	tempV = glm::rotateZ(tempV, zAngle);
	this->vVec = glm::normalize(tempV);

	//seDimensions
	dimensions = glm::vec2(dims.x, dims.y);

	//create temporary copies of U and V that are scaled by half dimensions, to get bounds of plane
	//	also create temp copy of normal in order to represent small depth and avoid errors with bounding box intersection
	glm::vec2 halfDims = dimensions / 2.0f;
	glm::vec3 uTemp = uVec * halfDims.x;
	glm::vec3 vTemp = vVec * halfDims.y;
	glm::vec3 nTemp = normal * PLANE_DEPTH;

	aabb->minBounds = position - (uTemp + vTemp + nTemp);
	aabb->maxBounds = position + (uTemp + vTemp + nTemp);
}

CUDA_DEVICE CUDA_HOST glm::vec2 Plane::getDimensions() const
{
	return dimensions;
}

CUDA_DEVICE bool Plane::intersects(Ray& ray, float& thit0, float& thit1) const
{
	float val = glm::dot(ray.dir, this->normal);

	//To intersect the plane the ray direction must point opposite to the normal
	if(val < 0)
	{
		//Point on Ray = origin + direction * t
		//point of intersection = origin + direction * thit
		//Plane: 0 = (point - position) � normal
		//	Any line on plane must be at 90 degrees to the plane's normal
		//	therefore dot product between line and normal is 0
		//Substitute point on ray for point in plane equation
		//	0 = (origin + dt - position) � normal
		//	0 = (origin - position) � normal + normal � (direction * t)
		// -normal � direction * t = (origin - position) � normal
		// t = -((origin - position) � normal) / (normal � direction)
		// t = ((position - origin) � normal) / (normal � direction)
		float temp = -glm::dot((ray.pos-position), this->normal) / val;

		//position of intersection using ray equation
		glm::vec3 intersection = ray.pos + (ray.dir * temp);

		//vector from plane position to intersection
		glm::vec3 toIntersect = intersection - this->position;

		//projection of toIntersect onto U, must have a magnitude less than half dimensions.x
		glm::vec3 intOnU = glm::dot(glm::normalize(toIntersect), uVec) * uVec;
		//projection of toIntersect onto V, must have a magnitude less than half dimensions.y
		glm::vec3 intOnV = glm::dot(glm::normalize(toIntersect), vVec) * vVec;

		if(glm::length(intOnU) < dimensions.x / 2.0f && glm::length(intOnV) < dimensions.y / 2.0f)
		{
			thit0 = thit1 = temp;
			return true;
		}
	}
	return false;
}

CUDA_DEVICE glm::vec3 Plane::calcWorldIntersectionNormal(const Ray& ray) const
{
	return this->normal;
}


Plane::~Plane()
{
}
