#include "Renderer.h"
#include "Math.h"


Renderer::Renderer(std::vector<Sphere> spheres, Image* i)
	:sphereList(spheres), image(i), camera()
{
}

Renderer::Renderer()
	: camera()
{
}


Renderer::~Renderer()
{
}

//returns if ray intersects sphere
//point p0 is closer to ray origin then p1
bool Renderer::testSphere(Ray ray, Sphere sphere, float& p0, float& p1)
{
	glm::vec3 val = (ray.pos - sphere.pos);
	float t = glm::dot(ray.dir, val);
	//test for intersections
	if(!Math::solveQuadratic(1.0f, 2.0f*glm::dot(ray.dir, val), glm::dot(val, val) - sphere.radius*sphere.radius, p0, p1))
		return false;
	//make x0 the closer point
	if(p1<p0) std::swap(p0, p1);
	return true;
}

void Renderer::render()
{
	if(image == nullptr) return;
	if(sphereList.size() == 0) return;
	float x, y;
	x = y = 0;
	Ray primary;
	camera.calculate(image->getAR());

	bool debugBreak = false;
	//traverse image columns
	for(unsigned i = 0; i < image->height; ++i)
	{
		//normalize y
		y = 1-(2 * (i+.5f) / image->height);
		//traverse rows
		for(unsigned j = 0; j < image->width; ++j)
		{
			if(i == 231 && j == 430)
				debugBreak = true;
			//normalize x
			x = (2 * (j+.5f) / image->width)-1;
			//x*right in terms of fov = x in terms of right axis
			primary.dir = glm::vec3(float(x)*image->getAR()*tan(camera.fov/2), float(y)*tan(camera.fov/2), -1) - primary.pos;
			primary.dir = glm::normalize(primary.dir);
			float minCollision = camera.viewDistance;
			//Check collision...generate shadow rays
			for(unsigned s = 0; s < sphereList.size(); ++s)
			{
				float p0, p1;
				p0 = p1 = 0;
				//test collision, implicit point on ray stored in p0 and p1
				if(testSphere(primary, sphereList[s], p0, p1))
				{
					if(minCollision < p0 || p0 < 0) break;
					minCollision = p0;
					Ray shadowRay;
					shadowRay.pos = primary.pos + primary.dir*p0;
					glm::vec3 finalCol;
					//iterate lights
					for(auto light : lightList)
					{
						glm::vec3 normal = shadowRay.pos - sphereList[s].pos;
						normal = glm::normalize(normal);
						shadowRay.dir = light.pos - shadowRay.pos;
						shadowRay.dir = glm::normalize(shadowRay.dir);
						//if pointing opposite directions skip light
						//if(glm::dot(normal, -shadowRay.dir) <= 0)
							//continue;
						bool inShadow = false;
						//check if in shadow
						for(unsigned s2 = 0; s2 < sphereList.size(); ++s2)
						{
							if(s2 == s) continue;
							float temp0, temp1;
							if(testSphere(shadowRay, sphereList[s2], temp0, temp1))
							{
								//inShadow = true;
							}
						}
						if(inShadow)
							finalCol = glm::vec3(0, 0, 0);
						else
						{
							float c = glm::dot(normal, shadowRay.dir);
							
							finalCol = sphereList[s].color *glm::dot(normal, shadowRay.dir); //* light.intensity;	
							if(c < 0)
							{
								finalCol = glm::vec3(0, 0, 0);
							}
							finalCol += ambientColor*ambientIntensity;
						}
						image->data[i*image->width+j] = finalCol;
					}
				}
			}
		}
	}
}
