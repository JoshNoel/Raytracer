#include "Renderer.h"
void main(char** args)
{
	Renderer renderer;

	Sphere sphere(glm::vec3(0, 0, -3), 1, glm::vec3(57, 166, 213));
	renderer.addSphere(sphere);
	Sphere sphere2(glm::vec3(0.9, 0.5, -4), 1, glm::vec3(200, 0, 0));
	renderer.addSphere(sphere2);
	Light light;
	light.pos = glm::vec3(-1.f, 4.0f, -3.0f);
	light.intensity = 1.0f;
	renderer.addLight(light);

	Image image(800, 600);
	renderer.image = &image;

	renderer.render();

	image.output("./OutputImage.ppm");
}