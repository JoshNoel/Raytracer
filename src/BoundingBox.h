#pragma once
#include "glm/glm.hpp"
#include "Ray.h"

class BoundingBox
{
public:
	BoundingBox();
	BoundingBox(glm::vec3 minBounds, glm::vec3 maxBounds);
	~BoundingBox();

	bool intersects(const Ray& ray, float& thit0, float& thit1) const;
	bool intersects(const Ray& ray) const;

	//x=0, y=1, z=2
	int getLongestAxis() const;

	//joins the minBounds and maxBounds of this* with bbox
	void join(const BoundingBox& bbox);

	glm::vec3 getCentroid() const;

	//minBounds(defaults to FLOAT_MAX) = smallest x value, smallest y value, most positive z value
	//maxBounds(defaults to negative FLOAT_MAX) = greatest x value, greatest y value, most negative z value
	glm::vec3 minBounds, maxBounds;
};
