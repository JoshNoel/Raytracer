#include "Triangle.h"
#include "glm\gtx\transform.hpp"
#include "TriObject.h"
#include "GeometryObj.h"


Triangle::Triangle(const std::array<glm::vec3, 3>& p, bool calcNormal)
	: points(p), Shape(getWorldPos())
{
	if(calcNormal)
		normal = calcObjectNormal();
	else
		normal = glm::vec3(0, 1, 0);
}

Triangle::~Triangle()
{
}

void Triangle::setUVCoords(const std::array<glm::vec2, 3>& uv)
{
	uvCoords = uv;
	hasUV = true;
}

std::array<glm::vec3, 3> Triangle::getPoints() const
{
	return points;
}

bool Triangle::getUV(std::array<glm::vec2, 3>& coords) const
{
	if(hasUV)
	{
		coords = this->uvCoords;
		return true;
	}
	else
	{
		return false;
	} 
}

//used to initialize
glm::vec3 Triangle::calcObjectNormal() const
{
	return glm::normalize(glm::cross(points[1] - points[0], points[2] - points[0]));
}

glm::vec3 Triangle::calcWorldIntersectionNormal(const Ray& ray) const
{
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
	glm::vec3 normal = calcWorldIntersectionNormal(Ray());
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
	float t = -(x / y);

	if(t < 0)
		return false;

	glm::vec3 intersection = ray.pos + ray.dir*t;

	if(glm::dot(glm::cross(points[1] - points[0], intersection - points[0]), normal) > 0)
	{
		if(glm::dot(glm::cross(points[2] - points[1], intersection - points[1]), normal) > 0)
		{
			if(glm::dot(glm::cross(points[0] - points[2], intersection - points[2]), normal) > 0)
			{
				if(t < thit0)
				{
					thit0 = t;
					if(t > thit1)
						thit1 = t;
					return true;
				}
			}

		}
	}

	return false;
}

