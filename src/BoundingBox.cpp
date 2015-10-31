#include "BoundingBox.h"
#include "MathHelper.h"

BoundingBox::BoundingBox(glm::vec3 minB, glm::vec3 maxB)
	: minBounds(minB), maxBounds(maxB)
{
}

BoundingBox::BoundingBox()
	: BoundingBox(glm::vec3(-_INFINITY, -_INFINITY, _INFINITY), glm::vec3(_INFINITY, _INFINITY, -_INFINITY))
{
}

BoundingBox::~BoundingBox()
{
}

void BoundingBox::join(const BoundingBox& bbox)
{
	//join minBounds
	if(minBounds.x > bbox.minBounds.x)
		minBounds.x = bbox.minBounds.x;
	if(minBounds.y > bbox.minBounds.y)
		minBounds.y = bbox.minBounds.y;
	//maxBounds.z < minBounds.z because -z away from camera direction
	if(minBounds.z < bbox.minBounds.z)
		minBounds.z = bbox.minBounds.z;

	//join maxBounds
	if(maxBounds.x < bbox.maxBounds.x)
		maxBounds.x = bbox.maxBounds.x;
	if(maxBounds.y < bbox.maxBounds.y)
		maxBounds.y = bbox.maxBounds.y;
	//maxBounds.z < minBounds.z because -z away from camera direction
	if(maxBounds.z > bbox.maxBounds.z)
		maxBounds.z = bbox.maxBounds.z;
}

int BoundingBox::getLongestAxis() const
{
	float xLength = maxBounds.x - minBounds.x;
	float yLength = maxBounds.y - minBounds.y;
	//lesser z is maxBound because -z is into scene, therefore lesser number = farther from camera
	float zLength = minBounds.z - maxBounds.z;

	if(xLength - yLength < 0)
	{
		if(yLength - zLength < 0)
			return 2;
		else
			return 1;
	}

	return 0;
}

glm::vec3 BoundingBox::getCentroid() const
{
	return glm::vec3((minBounds + maxBounds) / 2.0f);
}

bool BoundingBox::intersects(const Ray& ray, float& thit0, float& thit1) const
{
	//Ray = O+dt
	//Bmin = box minimum bounds on each axis
	//Bmax = box maximum bounds on each axis
	//tmin = t value of ray when it hits the aabb
	//tmax = t value of ray when it exits the aabb
	//O(x,y,z)+d(x,y,z)tmin(x,y,z) = Bmin(x,y,z) <-- intersection of a function with a line(at what input, t, does the ray's respective coordinate match the box's minimum coordinate
	//tmin = (Bmin-O)/d

	/*		   -x	   +x
				|		|
	  dir.x > 0	|		|	dir.x < 0	
	----------->|		|<-------------
				|		|
				|		|
	*/
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

	/*    minX			   maxX
		   |				|
		 1 |				|	2
  maxY ____|________________|______
		   |				|
		   |	 Inside		|
  minY ____|________________|______
		   |				|
		 3 |				|	4
		   |				|
	*/
	//if tymax < txmin then ray misses
	//It clips the top left corner (1) outside the box
	if(tmaxY < tminX)
		return false;

	//if txmax < tymin then ray misses
	//It clips the bottom right corner (4) outside the box
	if(tmaxX < tminY)
		return false;
	 
	float t0 = (tminX > tminY) ? tminX : tminY;
	float t1 = (tmaxX < tmaxY) ? tmaxX : tmaxY;

	//if ray is coming from front(dir.z < 0) then positive z is min
	//if ray is coming from back(dir.z > 0) then negative z is min
	float tminZ, tmaxZ;
	if(ray.dir.z < 0)
	{
		tmaxZ = (maxBounds.z - ray.pos.z) / ray.dir.z;
		tminZ = (minBounds.z - ray.pos.z) / ray.dir.z;
	}
	else
	{
		tmaxZ = (minBounds.z - ray.pos.z) / ray.dir.z;
		tminZ = (maxBounds.z - ray.pos.z) / ray.dir.z;
	}

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

	thit0 = t0;
	thit1 = t1;
	return true;
}

bool BoundingBox::intersects(const Ray& ray) const
{
	float t0, t1;
	return intersects(ray, t0, t1);
}
