#include "Renderer.h"
#include "MathHelper.h"
#include <iostream>
#include "TriObject.h"
#include "GeometryObj.h"

Renderer::Renderer(const Scene* s, Image* i)
	:image(i), camera(), scene(s)
{
	
}

Renderer::~Renderer()
{
}

void Renderer::render()
{
	if(image == nullptr) return;
	if(scene->objectList.size() == 0) return;
	float x, y;
	x = y = 0;
	Ray primary;
	//primary.pos(0, 0, camera.focalLength);
	camera.calculate(image->getAR());

	bool debugBreak = false;
	//traverse image columns
	for(int i = 0; i < image->height; ++i)
	{
		//normalize y
		//y = (2 * (i+.5f) / image->height)-1;
		y = 1-(2*(i+.5f) / image->height);
		//traverse rows
		for(int j = 0; j < image->width; ++j)
		{
			//normalize x
			x = (2*(j+.5f) / image->width) - 1.f;

			//x*right in terms of fov == x in terms of right axis
			//primary.dir = glm::vec3(float(x)*image->getAR()*tan(camera.fov/2), float(y)*tan(camera.fov/2), -1) - primary.pos;
			primary.dir = x*camera.right + y*camera.up + camera.direction;
			primary.dir = glm::normalize(primary.dir);

			//Check collision...generate shadow rays
			primary.thit = camera.viewDistance;

			//TRAVERSE KD TREE HERE

			for(unsigned s = 0; s < scene->objectList.size(); ++s)
			{
				/*if(scene->objectList[s]->getType() != Object::PLANE)
				{
					if(!scene->objectList[s]->aabb.intersects(primary))
						continue;
				}*/
				float p0, p1;
				p0 = _INFINITY;
				p1 = -_INFINITY;

				//test collision, p0 is tmin and p1 is tmax where collisions occur on ray
				if(scene->objectList[s]->getShape()->aabb.intersects(primary))
				{
					if(scene->objectList[s]->getShape()->intersects(primary, &p0, &p1))
					{
						//if p0 is not minimum, or is behind origin than this intersection is behind another object
						//in the rays path, or is behind the camera
						if(p0 > primary.thit || p0 < 0)
							continue;
						primary.thit = p0;

						//Create shadow ray at intersection to check if point is in a shadow (there is another object between
						//the point and a light
						Ray shadowRay;
						shadowRay.pos = primary.pos + primary.dir*p0;
						glm::vec3 finalCol;

						finalCol = scene->objectList[s]->getMaterial().color;
						//iterate lights
						/*for(auto light : scene->lightList)
						{
							glm::vec3 normal = scene->objectList[s]->getShape()->calcWorldIntersectionNormal(shadowRay.pos);
							normal = glm::normalize(normal);

							//Create shadow ray from intersection point to light
							shadowRay.dir = light.pos - shadowRay.pos;
							shadowRay.dir = glm::normalize(shadowRay.dir);*/
							//if pointing opposite directions skip light
							/*if(glm::dot(normal, shadowRay.dir) <= 0)
								continue;*/
							/*bool inShadow = false;
							//check if in shadow
							for(unsigned s2 = 0; s2 < scene->objectList.size(); ++s2)
							{
								if(s2 == s) continue;
								float temp0, temp1;
								temp0 = SHADOW_RAY_LENGTH;
								temp1 = -SHADOW_RAY_LENGTH;
								if(scene->objectList[s2]->getShape()->intersects(shadowRay, &temp0, &temp1))
								{
									//if(temp0 > 0)
									inShadow = true;
								}
							}
							if(inShadow)
								finalCol = glm::vec3(0, 0, 0);
							else
							{
								finalCol = scene->objectList[s]->getMaterial().color;*/ // * glm::max(0.0f, glm::dot(normal, shadowRay.dir)); //* light.intensity;	
								/*if(c < 0)
								{
									finalCol = glm::vec3(0, 0, 0);
								}*/
							//}
							finalCol += scene->ambientColor*scene->ambientIntensity;
							finalCol = glm::min(finalCol, glm::vec3(255, 255, 255));
							//finalCol = scene->objectList[s]->getMaterial().color;
							image->data[i*image->width + j] = finalCol;
						//}
					}
				}
			}
		}
	}
}
