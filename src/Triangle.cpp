#include "Triangle.h"
#include "glm\gtx\transform.hpp"
#include "TriObject.h"
#include "GeometryObj.h"


Triangle::Triangle(const std::array<glm::vec3, 3>& p)
	: points(p), Shape(getWorldPos())
{
}

Triangle::~Triangle()
{
}

glm::vec3 Triangle::calcObjectNormal() const
{
	return glm::normalize(glm::cross(points[1] - points[0], points[2] - points[0]));
}

glm::vec3 Triangle::calcWorldIntersectionNormal(glm::vec3) const
{
	std::array<glm::vec3, 3> p = getWorldCoords();
	return glm::normalize(glm::cross(p[1] - p[0], p[2] - p[0]));
}

std::array<glm::vec3, 3> Triangle::getWorldCoords() const
{
	std::array<glm::vec3, 3> p;
	for(unsigned int i = 0; i < 3; ++i)
	{
		glm::vec4 v = glm::vec4(points[i], 1);
		p[i] = glm::vec3(glm::translate(position) * v);
	}
	return p;
}

glm::vec3 Triangle::getWorldPos() const
{
	return position + glm::vec3((points[0] + points[1] + points[2]) / 3.0f);
}

bool Triangle::intersects(Ray& ray, float* thit0, float* thit1) const
{
	glm::vec3 normal = calcWorldIntersectionNormal(glm::vec3());
	glm::vec3 position = getWorldPos();	 
	std::array<glm::vec3, 3> points = getWorldCoords();

	float y = glm::dot(normal, ray.dir);
	//ray is parrellel to plane of triangle
	if(y < 1e-9)
		return false;
	float x = glm::dot(normal, position - ray.pos);
	float t = (x / y);

	glm::vec3 intersection = ray.pos + ray.dir*t;

	if(glm::dot(glm::cross(points[1] - points[0], intersection - points[0]), normal) > 0)
	{
		if(glm::dot(glm::cross(points[2] - points[1], intersection - points[1]), normal) > 0)
		{
			if(glm::dot(glm::cross(points[0] - points[2], intersection - points[2]), normal) > 0)
			{
				if(t < ray.thit)
				{
					ray.thit = t;
				}

				if(t < *thit0)
				{
					*thit0 = *thit1 = t;
					return true;
				}
			}

		}
	}

	return false;
}

