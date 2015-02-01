#include "Camera.h"


Camera::Camera()
	: position(0, 0, 0),
	direction(0, 0, -1),
	up(0, 1, 0),
	fov(90.f),
	viewDistance(100.f)
{
}


Camera::~Camera()
{
}

void Camera::calculate(float aspectRatio)
{
	glm::normalize(up);
	glm::normalize(direction);

	float fovRad = fov*3.1415 / 180;

	float thfov = tan(fovRad / 2);

	right = glm::cross(direction, up);
	glm::normalize(right);

	right *= thfov * aspectRatio;
	up *= thfov;
}
