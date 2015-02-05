#pragma once
#include "glm\glm.hpp"

class Camera
{
public:
	Camera();
	~Camera();

	glm::vec3 up, direction, right, position;
	float fov;
	float viewDistance;
	float focalLength;

	inline void setPosition(glm::vec3 v)
	{ position = v; }
	inline void setUp(glm::vec3 v){ up = v; }
	inline void lookAt(glm::vec3 point){ direction = point - position; }
	void calculate(float aspectRatio);
	};

