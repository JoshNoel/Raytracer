#include "Renderer.h"
#include "Math.h"
#include <iostream>


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

//returns if ray intersects object
//point p0 is closer to ray origin then p1
bool Renderer::testObject(const Ray& ray, Object* object, float& p0, float& p1)
{
	switch(object->getType())
	{
	case Object::OBJECT_TYPE::SPHERE:
	{
		Sphere* sphere = dynamic_cast<Sphere*>(object);
		glm::vec3 val = (ray.pos - object->position);
		float t = glm::dot(ray.dir, val);
		//test for intersections
		if(!Math::solveQuadratic(1.0f, 2.0f*glm::dot(ray.dir, val), glm::dot(val, val) - sphere->radius*sphere->radius, p0, p1))
			return false;
		//make x0 the closer point
		if(p1<p0) std::swap(p0, p1);
		return true;
	}
	case Object::OBJECT_TYPE::PLANE:
	{
		Plane* plane = dynamic_cast<Plane*>(object);
		float val = glm::dot(ray.dir, plane->normal);
		if(val != 0)
		{
			p0 = p1 = glm::dot((plane->position - ray.pos), plane->normal) / val;
			//if(p0 > 0)
				//std::cout << "intersection/n";
			return (p0 > 0);
		}
	}
	default:
		return false;
		break;
	}	
}

glm::vec3 Renderer::calcNormal(Object* object, glm::vec3 p0)
{
	switch(object->getType())
	{
	case Object::OBJECT_TYPE::SPHERE:
	{
		return p0 - object->position;
	}
	case Object::OBJECT_TYPE::PLANE:
	{
		Plane* plane = dynamic_cast<Plane*>(object);
		return plane->normal;
	}
	default:
		break;
	}
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
			float minCollision = camera.viewDistance;
			//Check collision...generate shadow rays
			for(unsigned s = 0; s < objectList.size(); ++s)
			{
				float p0, p1;
				p0 = p1 = 0;
				//test collision, implicit point on ray stored in p0 and p1
				if(testObject(primary, objectList[s].get(), p0, p1))
				{
					if(minCollision < p0 || p0 < 0) break;
					minCollision = p0;
					Ray shadowRay;
					shadowRay.pos = primary.pos + primary.dir*p0;
					glm::vec3 finalCol;
					//iterate lights
					for(auto light : lightList)
					{
						glm::vec3 normal = calcNormal(objectList[s].get(), shadowRay.pos);
						normal = glm::normalize(normal);
						shadowRay.dir = light.pos - shadowRay.pos;
						shadowRay.dir = glm::normalize(shadowRay.dir);
						//if pointing opposite directions skip light
						//if(glm::dot(normal, shadowRay.dir) <= 0)
							//continue;
						bool inShadow = false;
						//check if in shadow
						for(unsigned s2 = 0; s2 < objectList.size(); ++s2)
						{
							if(s2 == s) continue;
							float temp0, temp1;
							if(testObject(shadowRay, objectList[s2].get(), temp0, temp1))
							{
								if(temp0 > 0)
									inShadow = true;
							}
						}
						if(inShadow)
							finalCol = glm::vec3(0, 0, 0);
						else
						{
							float c = glm::dot(normal, shadowRay.dir);
							
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
