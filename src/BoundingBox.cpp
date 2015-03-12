#include "BoundingBox.h"
#include <limits>
#include <utility>
#define AABB_MAX 10000

BoundingBox::BoundingBox(glm::vec3 minB, glm::vec3 maxB)
	: minBounds(minB), maxBounds(maxB)
{
}

BoundingBox::BoundingBox()
	: BoundingBox(glm::vec3(-AABB_MAX, -AABB_MAX, AABB_MAX), glm::vec3(AABB_MAX, AABB_MAX, -AABB_MAX))
{
}

BoundingBox::~BoundingBox()
{
}

bool BoundingBox::intersects(const Ray ray) const
{
	//Ray = O+dt
	//Bmin = box minimum bounds on each axis
	//Bmax = box maximum bounds on each axis
	//tmin = t value of ray when it hits the aabb
	//tmax = t value of ray when it exits the aabb
	//O(x,y,z)+d(x,y,z)tmin(x,y,z) = Bmin(x,y,z) <-- intersection of a function with a line(at what input, t, does the ray's respective coordinate match the box's minimum coordinate
	//tmin = (Bmin-O)/d

	//if ray is coming from left(dir.x > 0) then negative x is min
	//if ray is coming from right(dir.x < 0) then positive x is min
	float tminX, tmaxX;
	if(ray.dir.x >= 0)
	{
		tminX = (minBounds.x - ray.pos.x) / ray.dir.x;
		tmaxX = (maxBounds.x - ray.pos.x) / ray.dir.x;
	}
	else
	{
		tminX = (maxBounds.x - ray.pos.x) / ray.dir.x;
		tmaxX = (minBounds.x - ray.pos.x) / ray.dir.x;
	}

	//if ray is coming from bottom(dir.y > 0) then negative y is min
	//if ray is coming from top(dir.y < 0) then positive y is min
	float tminY, tmaxY;
	if(ray.dir.y > 0)
	{
		tmaxY = (maxBounds.y - ray.pos.y) / ray.dir.y;
		tminY = (minBounds.y - ray.pos.y) / ray.dir.y;
	}
	else
	{
		tmaxY = (minBounds.y - ray.pos.y) / ray.dir.y;
		tminY = (maxBounds.y - ray.pos.y) / ray.dir.y;
	}

	//if tymax < txmin then ray misses
	if(tmaxY < tminX)
		return false;

	//if txmax < tymin then ray misses
	if(tmaxX < tminY)
		return false;
	 
	float t0 = (tminX > tminY) ? tminX : tminY;
	float t1 = (tmaxX < tmaxY) ? tmaxX : tmaxY;

	float tminZ = (minBounds.z - ray.pos.z) / ray.dir.z;
	float tmaxZ = (maxBounds.z - ray.pos.z) / ray.dir.z;

	//if ray enters the x and y bounds of the aabb after the maximimum z, it missed the box behind
	if(t0 > tmaxZ)
		return false;
	//if ray exits the x and y bounds of the aabb before the minimum z then it missed in front
	if(t1 < tminZ)
		return false;

	if(tminZ > t0)
		t0 = tminZ;
	if(tmaxZ < t1)
		t1 = tmaxZ;
	/*if(t0 < 0 || t1 < 0)
		return false;*/

	return true;
}
