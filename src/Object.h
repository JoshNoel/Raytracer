#pragma once
#include "Material.h"
#include "glm\glm.hpp"
#include "Ray.h"
#include "BoundingBox.h"

class Object
{
public:
	glm::vec3 position;
	inline Material& getMaterial() { return material; }

	enum OBJECT_TYPE
	{
		TRIANGLE_BASED = 0,
		SPHERE,
		PLANE		
	};

	virtual OBJECT_TYPE getType() const = 0;
	virtual ~Object();

	virtual bool intersects(const Ray ray, float& t0, float& t1) const = 0;
	virtual glm::vec3 calcNormal(glm::vec3 p0) const = 0;
	BoundingBox aabb;

protected:
	Object(glm::vec3, Material);
	Object();

private:
	Material material;
	
};

