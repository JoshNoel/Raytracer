#pragma once
#include "CudaDef.h"
#include "glm/glm.hpp"
#include "managed.h"

class Camera : public Managed
{
public:
	Camera();
	~Camera();

	glm::vec3 up, direction, right, position;

	//vertical FOV
	float fov;
	float viewDistance;
	float focalLength;

	//sets camera position
	inline void setPosition(glm::vec3 v)
	{ position = v; }

	//sets the up vector of the camera
	inline void setUp(glm::vec3 v){ up = v; }

	//calculates direction of camera using a point to look at
	inline void lookAt(glm::vec3 point){ direction = point - position; }

	//calculates the camera's properties for final use in Renderer
	void calculate(float aspectRatio);
	};

