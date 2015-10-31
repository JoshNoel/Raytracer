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
	glm::vec3 normal = glm::normalize(glm::cross(p[1] - p[0], p[2] - p[0]));
	return normal;
}

std::array<glm::vec3, 3> Triangle::getWorldCoords() const
{
	std::array<glm::vec3, 3> p;
	for(unsigned int i = 0; i < 3; ++i)
	{
		glm::vec4 v = glm::vec4(points[i], 1);
		p[i] = glm::vec3((glm::translate(glm::mat4(1.0f), position) * v));
	}
	return p;
}

glm::vec3 Triangle::getWorldPos() const
{
	return position + glm::vec3((points[0] + points[1] + points[2]) / 3.0f);
}

bool Triangle::intersects(Ray& ray, float& thit0, float& thit1) const
{
	glm::vec3 normal = calcWorldIntersectionNormal(glm::vec3());
	glm::vec3 position = getWorldPos();	 
	std::array<glm::vec3, 3> points = getWorldCoords();

	//Ray-Triangle intersection: -[((o-P₀) • N)]/[d • N] = t

	float y = glm::dot(normal, ray.dir);

	//ray points same direction as normal, so if it intersects it hits the back of the triangle
	if(y > 0)
		return false;

	//ray is parrellel to plane of triangle
	if(y > -1e-6)
		return false;

	//else ray intersects plane of triangle

	float x = glm::dot(normal, ray.pos - position);
	float t = (-x / y);

	glm::vec3 intersection = ray.pos + ray.dir*t;

	if(glm::dot(glm::cross(points[1] - points[0], intersection - points[0]), normal) > 0)
	{
		if(glm::dot(glm::cross(points[2] - points[1], intersection - points[1]), normal) > 0)
		{
			if(glm::dot(glm::cross(points[0] - points[2], intersection - points[2]), normal) > 0)
			{
				if(t < thit0)
				{
					thit0 = thit1 = t;
					return true;
				}
			}

		}
	}

	return false;
}

