#pragma once
#include "Material.h"
#include "glm\glm.hpp"
class Object
{
public:
	glm::vec3 position;
	inline Material& getMaterial() { return material; }

	enum OBJECT_TYPE
	{
		SPHERE = 0,
		PLANE
	};

	virtual OBJECT_TYPE getType() const = 0;
	virtual ~Object();


protected:
	Object(glm::vec3, Material);
	Object();

private:
	Material material;
};

