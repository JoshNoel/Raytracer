#include "Renderer.h"
#include "MathHelper.h"
#include <iostream>
#include "TriObject.h"

Renderer::Renderer(std::vector<std::unique_ptr<Object>> objects, Image* i)
	:image(i), camera()
{
	objectList = std::move(objects);
}

Renderer::Renderer()
	: camera()
{
}


Renderer::~Renderer()
{
}

void Renderer::render()
{
	if(image == nullptr) return;
	if(objectList.size() == 0) return;
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
			if(i == 231 && j == 430)
				debugBreak = true;
			//normalize x
			//x = 1-(2 * (j+.5f) / image->width);
			x = (2*(j+.5f) / image->width) - 1.f;
			//x*right in terms of fov = x in terms of right axis
			//primary.dir = glm::vec3(float(x)*image->getAR()*tan(camera.fov/2), float(y)*tan(camera.fov/2), -1) - primary.pos;
			primary.dir = x*camera.right + y*camera.up + camera.direction;
			primary.dir = glm::normalize(primary.dir);
			//Check collision...generate shadow rays
			float minCollision = camera.viewDistance;
			for(unsigned s = 0; s < objectList.size(); ++s)
			{
				if(s == 0 && i > 700)
					int k = 2;
				if(objectList[s]->getType() != Object::PLANE)
				{
					if(!objectList[s]->aabb.intersects(primary))
						continue;
				}
				float p0, p1;
				p0 = p1 = minCollision;
				//test collision, implicit point on ray stored in p0 and p1
				if(objectList[s]->intersects(primary, p0, p1))
				{
					if(p0 > minCollision || p0 < 0)
						continue;
					minCollision = p0;
					Ray shadowRay;
					shadowRay.pos = primary.pos + primary.dir*p0;
					glm::vec3 finalCol;

					
					//iterate lights
					for(auto light : lightList)
					{	
						glm::vec3 normal = objectList[s]->calcNormal(shadowRay.pos);
						normal = glm::normalize(normal);
						shadowRay.dir = light.pos - shadowRay.pos;
						shadowRay.dir = glm::normalize(shadowRay.dir);
						//if pointing opposite directions skip light
						/*if(glm::dot(normal, shadowRay.dir) <= 0)
							continue;*/
						bool inShadow = false;
						//check if in shadow
						for(unsigned s2 = 0; s2 < objectList.size(); ++s2)
						{
							if(s2 == s) continue;
							float temp0, temp1;
							temp0 = temp1 = SHADOW_RAY_LENGTH;
							if(objectList[s2]->intersects(shadowRay, temp0, temp1))
							{
								//if(temp0 > 0)
									inShadow = true;
							}
						}
						if(inShadow)
							finalCol = glm::vec3(0, 0, 0);
						else
						{						
							finalCol = objectList[s]->getMaterial().color * glm::max(0.0f, glm::dot(normal, shadowRay.dir)); //* light.intensity;	
							/*if(c < 0)
							{
								finalCol = glm::vec3(0, 0, 0);
							}*/
						}
						finalCol += ambientColor*ambientIntensity;
						finalCol = glm::min(finalCol, glm::vec3(255, 255, 255));
						//finalCol = objectList[s]->getMaterial().color;
						image->data[i*image->width+j] = finalCol;
					}
				}
			}
		}
	}
}
