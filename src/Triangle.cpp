#include "Triangle.h"
#include "TriObject.h"
#include "GeometryObj.h"

const int Triangle::parameters::PARAM_SIZES[Triangle::parameters::MAX_PARAMS] = {
0,
sizeof(helper::array<glm::vec3, 3>),
sizeof(helper::array<glm::vec3, 3>) + sizeof(bool),
sizeof(helper::array<glm::vec3, 3>) + sizeof(bool) + sizeof(glm::vec3),
sizeof(helper::array<glm::vec3, 3>) + sizeof(bool) + sizeof(glm::vec3) + sizeof(helper::array<glm::vec2, 3>),
sizeof(helper::array<glm::vec3, 3>) + sizeof(bool) + sizeof(glm::vec3) + sizeof(helper::array<glm::vec2, 3>) + sizeof(BoundingBox*)
};

/*
Triangle::parameters::parameters(const helper::array<glm::vec3, 3>& points, bool calc_normal, const glm::vec3& position, const helper::array<glm::vec2, 3>& uv_array, BoundingBox* aabb)
{
	data = new Data();
	data->points = points;
	data->calcNormal = calc_normal;
	data->position = position;
	data->uv_array = uv_array;
	data->aabb = aabb;
	num_params = 5;
}

Triangle::parameters::parameters(const helper::array<glm::vec3, 3>& points, bool calc_normal)
{
	data = new Data();
	data->points = points;
	data->calcNormal = calc_normal;

	helper::array<glm::vec2, 3> a;
	a[0] = glm::vec2();
	a[1] = glm::vec2();
	a[2] = glm::vec2();
	data->uv_array = a;
	data->aabb = new BoundingBox();
	num_params = 2;
}
*/
	
void Triangle::setUVCoords(const helper::array<glm::vec2, 3>& uv)
{
	*uvCoords = uv;
	hasUV = true;
}

CUDA_DEVICE CUDA_HOST helper::array<glm::vec3, 3, true> Triangle::getPoints() const
{
	return *points;
}

CUDA_DEVICE CUDA_HOST bool Triangle::getUV(helper::array<glm::vec2, 3, true>& coords) const
{
	if(hasUV)
	{
		coords = *this->uvCoords;
		return true;
	}
	else
	{
		return false;
	} 
}

CUDA_DEVICE CUDA_HOST glm::vec3 Triangle::calcObjectNormal() const
{
	return glm::normalize(glm::cross((*points)[1] - (*points)[0], (*points)[2] - (*points)[0]));
}

CUDA_DEVICE glm::vec3 Triangle::calcWorldIntersectionNormal(const Ray& ray) const
{
	return normal;
}

CUDA_DEVICE CUDA_HOST void Triangle::setPosition(const glm::vec3& pos)
{
	//updates points in world space and triangleCenter 
	position = pos;
	helper::array<glm::vec3, 3> p;
	glm::mat4 translation = glm::mat4(1.0f);
	translation[3] = glm::vec4(pos, 1);
	for(unsigned int i = 0; i < 3; ++i)
	{
		glm::vec4 v = glm::vec4((*points)[i], 1);
		p[i] = glm::vec3(translation * v);
	}

	*points_world = p;
	triangleCenter = calcCenter();
}

CUDA_DEVICE CUDA_HOST helper::array<glm::vec3, 3, true> Triangle::getWorldCoords() const
{
	return *points_world;
}

CUDA_DEVICE CUDA_HOST glm::vec3 Triangle::calcCenter() const
{
	return position + glm::vec3(((*points)[0] + (*points)[1] + (*points)[2]) / 3.0f);
}

CUDA_DEVICE bool Triangle::intersects(Ray& ray, float& thit0, float& thit1) const
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
	if(glm::dot(glm::cross((*points_world)[1] - (*points_world)[0], intersection - (*points_world)[0]), normal) > 0) {
		if(glm::dot(glm::cross((*points_world)[2] - (*points_world)[1], intersection - (*points_world)[1]), normal) > 0) {
			if(glm::dot(glm::cross((*points_world)[0] - (*points_world)[2], intersection - (*points_world)[2]), normal) > 0) {
					thit0 = t;
					if(t > thit1)
						thit1 = t;
					return true;
			}

		}
	}

	return false;
}

