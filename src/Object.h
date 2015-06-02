#pragma once
#include "Material.h"
#include "glm\glm.hpp"
#include "Ray.h"
#include "BoundingBox.h"

class Object
{
public:

	enum OBJECT_TYPE
	{
		GEOMETRY = 0,
		BVH
	};

	virtual OBJECT_TYPE getType() const = 0;
	virtual ~Object();

	virtual bool intersects(Ray& ray, float* thit) const = 0;
	

protected:
	Object();
	
};

