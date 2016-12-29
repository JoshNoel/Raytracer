#pragma once
#include "Ray.h"
#include "CudaDef.h"
#include "glm/glm.hpp"
#include "managed.h"

class BoundingBox 
	: public Managed
{
public:
	CUDA_DEVICE CUDA_HOST BoundingBox();
	CUDA_DEVICE CUDA_HOST BoundingBox(const BoundingBox&);
	CUDA_DEVICE CUDA_HOST BoundingBox(BoundingBox&&);

	CUDA_DEVICE CUDA_HOST BoundingBox(glm::vec3 minBounds, glm::vec3 maxBounds);
	CUDA_DEVICE CUDA_HOST ~BoundingBox();

	CUDA_DEVICE CUDA_HOST BoundingBox& operator=(const BoundingBox&);
	CUDA_DEVICE CUDA_HOST BoundingBox& operator=(BoundingBox&&);



	CUDA_DEVICE bool intersects(const Ray& ray, float& thit0, float& thit1) const;
	CUDA_DEVICE bool intersects(const Ray& ray) const;

	//x=0, y=1, z=2
	CUDA_DEVICE int getLongestAxis() const;

	//joins the minBounds and maxBounds of this* with bbox
	CUDA_DEVICE CUDA_HOST void join(const BoundingBox& bbox);

	CUDA_DEVICE CUDA_HOST glm::vec3 getCentroid() const;

	//minBounds(defaults to FLOAT_MAX) = smallest x value, smallest y value, most positive z value
	//maxBounds(defaults to negative FLOAT_MAX) = greatest x value, greatest y value, most negative z value
	glm::vec3 minBounds, maxBounds;
};
