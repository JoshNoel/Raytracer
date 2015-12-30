#include "Triangle.h"
#include "glm\gtx\transform.hpp"
#include "TriObject.h"
#include "GeometryObj.h"


Triangle::Triangle(const std::array<glm::vec3, 3>& p, bool calcNormal)
	: points(p), Shape(calcCenter())
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

glm::vec3 Triangle::calcObjectNormal() const
{
	return glm::normalize(glm::cross(points[1] - points[0], points[2] - points[0]));
}

glm::vec3 Triangle::calcWorldIntersectionNormal(const Ray& ray) const
{
	return normal;
}

void Triangle::setPosition(const glm::vec3& pos)
{
	//updates points in world space and triangleCenter 
	position = pos;
	std::array<glm::vec3, 3> p;
	for(unsigned int i = 0; i < 3; ++i)
	{
		glm::vec4 v = glm::vec4(points[i], 1);
		p[i] = glm::vec3((glm::translate(glm::mat4(1.0f), position) * v));
	}

	points_world = p;
	triangleCenter = calcCenter();
}

std::array<glm::vec3, 3> Triangle::getWorldCoords() const
{
	return points_world;
}

glm::vec3 Triangle::calcCenter() const
{
	return position + glm::vec3((points[0] + points[1] + points[2]) / 3.0f);
}

bool Triangle::intersects(Ray& ray, float& thit0, float& thit1) const
{	 
	//Ray-Triangle intersection: -[((o-P₀) • N)]/[d • N] = t

	float y = glm::dot(normal, ray.dir);

	//ray points same direction as normal, so if it intersects it hits the back of the triangle
	if(y > 0)
		return false;

	//ray is parrellel to plane of triangle
	if(y > -1e-6)
		return false;

	//else ray intersects plane of triangle
	float x = glm::dot(normal, ray.pos - triangleCenter);
	float t = -(x / y);

	if(t < 0)
		return false;

	//intersection will only be valid if the intersection is in front of the prvious intersection
		//(if there is one)
	if(t >= thit0)
	{
		return false;
	}

	glm::vec3 intersection = ray.pos + ray.dir*t;

	//checks if the ray intersects the intersection point is within the triangle formed by the three points
		//side vector = point on tri TO next point counterclockwise
		//to intersection vector = point on tri TO intersection point with the plane
		//if the ray intersects the triangle's plane outside the bounds of the triangle then
			//(side vector) X (to intersection vector) will point in opposite direction as the triangle's
			//normal for one of the points on the triangle
	if(glm::dot(glm::cross(points_world[1] - points_world[0], intersection - points_world[0]), normal) > 0)
	{
		if(glm::dot(glm::cross(points_world[2] - points_world[1], intersection - points_world[1]), normal) > 0)
		{
			if(glm::dot(glm::cross(points_world[0] - points_world[2], intersection - points_world[2]), normal) > 0)
			{
					thit0 = t;
					if(t > thit1)
						thit1 = t;
					return true;
			}

		}
	}

	return false;
}

