#include "Renderer.h"
#include "MathHelper.h"
#include <iostream>
#include "TriObject.h"
#include "GeometryObj.h"
#include "glm/gtx/rotate_vector.hpp"
#include <iterator>



const float Renderer::SHADOW_RAY_LENGTH = 50.0f;
const int Renderer::NUM_THREADS = 8;

Renderer::Renderer(const Scene* s, Image* i)
	:image(i), camera(), scene(s), distributionX(0, std::nextafter(1.0f, FLT_MAX)),
        distributionY(0, std::nextafter(1.0f, FLT_MAX)), rng(1), pixelsRendered(0)
{
	//initialize random number generator for shadow sampling
	std::random_device device;
	rng.seed(device());
}

/*
*snell's law: n1sin(theta1) = n2sin(theta2)
*When incident vector is a unit vector, snell's law
*relates the magnitudes of the perpendicular to normal component of the incident and
*refracted vectors, where the vectors in the ratio are coplanar with the incident and normal vectors
*n1*Iperp = n2*Rperp
*One can project I onto N
*	I•N*N
*Then get component perpendicular to N
*	I+(-I•N*N)
*This component is then related to the perpendicular component of the refracted vector through the ratio
*	n1/n2
*Therefore the perpendicular component of the reflected vector is
*	(n1/n2)*(I+(I•N*N))
*The magnitude of the parallel components of the incident and refracted vector must be equal in order to 
*maintain the ratio according to snell's law, however the direction is opposite
*Therefore R = -IncidentParallel + ReflectedPerpendicular
*	R = (I•N*N) + (n1/n2)*(I+(I•N*N))
*/
glm::vec3 refract(glm::vec3 dir, glm::vec3 norm, float ior1, float ior2)
{
	glm::vec3 dirOnNorm = glm::dot(-dir, norm)*norm;

	//sin(theta1) = |perpToNormal| because hypotonuse (the incident vector) has mag. of 1
	glm::vec3 iPerpToNorm = dir + dirOnNorm;
	float ratio = ior1 / ior2;
	glm::vec3 rPerpToNorm = ratio * iPerpToNorm;
	float thetaInner = std::asin(glm::length(rPerpToNorm));
	glm::vec3 rOnNorm = cosf(thetaInner) * -norm;

	return glm::normalize(rPerpToNorm + rOnNorm);
}


/////////Calculate Specular Component at Intersection//////////
/******REFLECTION RAY********
//	*    N
//	*	 ^
//	*	 |
//	*	 |
//	*  A Iₙ A
//	*^---^---^
//	* \	 |  /
//	*  \ϴ|ϴ/
//	*  I\|/R
//	*	 P
//	*
//	*R = (2Iₙ - I) - P
//	*Iₙ = I•N / |N|
//	*R = (2(I•N)/|N| - I) - P
//	*/
glm::vec3 reflect(glm::vec3 dir, glm::vec3 norm)
{
	return glm::normalize(dir - (2 * glm::dot(dir, norm) * norm));
}

bool Renderer::init()
{
	if(image == nullptr) return false;
	if(scene->objectList.size() == 0) return false;

	camera.calculate(image->getAR());

	return true;
}

//Runs code for each tread to render its section of the image
glm::vec3 Renderer::renderPixel(int pixelX, int pixelY) const
{
	int width = image->width;
	int samples = scene->PRIMARY_SAMPLES;
	glm::vec3 color;

	for(int primarySample = 0; primarySample < samples; ++primarySample)
	{
		float x, y;
		Ray primary;
		//normalize x
		x = (2.0f * (pixelX + distributionX(rng)) / image->width) - 1.0f;
		//normalize y
		y = 1.0f - (2.0f * (pixelY + distributionY(rng)) / image->height);

		primary.dir = x*camera.right + y*camera.up + camera.direction;
		primary.dir = glm::normalize(primary.dir);

		//Check collision...generate shadow rays
		primary.thit0 = camera.viewDistance;

		float thit0, thit1;
		color += glm::min(castRay(primary, thit0, thit1, 0), glm::vec3(255, 255, 255));
	}

	return (color / float(samples));
}

void Renderer::writeImage(glm::vec3 color, int x, int y) const
{
	image->data[x + y*image->width] = color;
}

bool Renderer::hitsObject(Ray& ray, float& thit0, float& thit1) const
{
	bool hit = false;
	for(unsigned s = 0; s < scene->objectList.size(); ++s)
	{

		float p0, p1;
		p0 = _INFINITY;
		p1 = -_INFINITY;

		//test collision, p0 is tmin and p1 is tmax where collisions occur on ray
		if(scene->objectList[s]->getShape()->aabb.intersects(ray))
		{
			if(scene->objectList[s]->getShape()->intersects(ray, p0, p1))
			{
				//if p0 is not minimum, or is behind origin than this intersection is behind another object
				//in the rays path, or is behind the camera
				if(p0 < ray.thit0 && p0 > 0)
				{
					ray.thit0 = p0;
					ray.thit1 = p1;
					ray.hitObject = scene->objectList[s].get();
					hit = true;
				}
			}
		}
	}

	return hit;
}

bool Renderer::hitsObject(Ray& ray) const
{
	float x, y;
	x =  _INFINITY;
	y = -_INFINITY;
	bool hit = hitsObject(ray, x, y);
	return (hit && x > 0);
}


glm::vec3 Renderer::castRay(Ray& ray, float& thit0, float& thit1, int depth) const
{
	glm::vec3 finalCol = glm::vec3(0,0,0);

	if(depth > scene->MAX_RECURSION_DEPTH)
	{
		finalCol = scene->bgColor;
	}
	else
	{
		if(!hitsObject(ray, thit0, thit1))
		{
			finalCol = scene->bgColor;
		}
		else
		{
			finalCol = glm::vec3(0,0,0);
			Ray shadowRay;
			shadowRay.pos = ray.pos + ray.dir * ray.thit0;

			glm::vec3 normal = ray.hitObject->getShape()->calcWorldIntersectionNormal(ray);
			normal = glm::normalize(normal);

			for(auto light : scene->lightList)
			{
				bool inShadow = true;

				//////Check visibility to light//////
				if(light.intensity < 0.0f)
					light.intensity = 0.0f;

				float lightVisibility = 0.0f;
				float lightFalloffIntensity = 0.0f;
				switch(light.type)
				{
					case Light::POINT:
					{
						//Monte Carlo Method - repeated random sampling of data approaches true value
							//Fire rays from intersection point to areaLight, and average results
								//samples are stratified over the area of the light
						if(light.isAreaLight && light.areaShape != nullptr)
						{
							float halfX = light.areaShape->getDimensions().x / 2.0f;
							float halfY = light.areaShape->getDimensions().y / 2.0f;
							std::uniform_real_distribution<float> distributionX(-1.0f, std::nextafter(1.0f, FLT_MAX));
							std::uniform_real_distribution<float> distributionY(-1.0f, std::nextafter(1.0f, FLT_MAX));

							//In order to create grid scene->SHADOW_SAMPLES must be a perfect square
							for(int sampleX = -scene->SQRT_DIV2_SHADOW_SAMPLES; sampleX < scene->SQRT_DIV2_SHADOW_SAMPLES; sampleX++)
							{
								for(int sampleY = -scene->SQRT_DIV2_SHADOW_SAMPLES; sampleY < scene->SQRT_DIV2_SHADOW_SAMPLES; sampleY++)
								{
									float xPos = distributionX(rng);
									float yPos = distributionY(rng);

									//change base position based on sample index to create grid of area SHADOW_SAMPLES
									//each random sample will be taken from square of grid to stratify the results
									//
									//numGridSquares = scene->SHADOW_SAMPLES;
									//numGridSquares MUST BE PERFECT SQUARE TO CREATE GRID;
									//1 grid square side length = halfDimOfPlane / (sqrt(numGridSquares)/2)
									//numGridSquaresPerSide = sqrt(numGridSquares)
									//
									//coordinate of center grid square at heightIndex h and widthIndex w,
									//when h and y have domain[-numGridSquaresPerSide/2, numGridSquaresPerSide/2] = 
									//	Center + [((w * gridSquareSideLength) - sign(w) * gridSquareSideLength / 2), 
									//				(h * gridSquareSideLength) - sign(h) * gridSquareSideLength / 2)
									float numGridSquares = scene->SHADOW_SAMPLES;
									float gridSquareSideLength = halfX / (std::sqrt(numGridSquares) / 2.0f);

									//center of grid square
									//	traverse each column
									glm::vec3 basePos = light.areaShape->getPosition() +
										(light.areaShape->getU() * ((sampleX * gridSquareSideLength) - (Math::sign(sampleX) * gridSquareSideLength / 2.0f))) +
										(light.areaShape->getV() * ((sampleY * gridSquareSideLength) - (Math::sign(sampleY) * gridSquareSideLength / 2.0f)));

									//add random vector to center of grid square
									//	this vector will give a random point within the grid square
									glm::vec3 randVec;
									randVec = ((gridSquareSideLength / 2.0f) * xPos * light.areaShape->getU()) +
										((gridSquareSideLength / 2.0f) * yPos * light.areaShape->getV());
									glm::vec3 randPosOnPlane = basePos + randVec;

									Ray toLight;
									//Ray starts at intersection point
									toLight.pos = shadowRay.pos;
									//Ray goes from intersection, to the randomly generated position
									toLight.dir = randPosOnPlane - shadowRay.pos;
									toLight.dir = glm::normalize(toLight.dir);
									//small displacement added along normal to avoid self-intersection
									toLight.pos += normal * RAY_EPSILON;
							
									//Light only contributes if it faces the object								
									if(glm::dot(-toLight.dir, light.areaShape->getNormal()) > 0.0f)
									{
										//Intersection point must also have a normal facing the light
										//if there is an object between the intersection and point on light, the point is in shadow
										if(glm::dot(normal, toLight.dir) > 0.0f && !hitsObject(toLight))
										{
											inShadow = false;
											lightVisibility++;
										}
									}
								}
							}

							//calculate final shadow ray for lighting calculations
							//also calculate light falloff using simple inverser square falloff
							shadowRay.dir = light.pos - shadowRay.pos;
							float lightRadius = glm::length(shadowRay.dir);
							shadowRay.dir = glm::normalize(shadowRay.dir);
							lightFalloffIntensity = (light.intensity) / (std::pow(lightRadius, 2.0f));
							lightVisibility /= scene->SHADOW_SAMPLES;
						} 
						else 
						{

							//if its not an area light, treat it as a simple point light with hard visible/not visible falloff
							shadowRay.dir = light.pos - shadowRay.pos;
							float lightRadius = glm::length(shadowRay.dir);
							shadowRay.dir = glm::normalize(shadowRay.dir);
							Ray temp = shadowRay;
							temp.pos += normal * RAY_EPSILON;
							if(!hitsObject(temp) && glm::dot(normal, shadowRay.dir) > 0.0f)
							{
								inShadow = false;
								lightFalloffIntensity = (light.intensity) / (4 * _PI_ * lightRadius);
							}
						}
						break;
					}
					case Light::DIRECTIONAL:
					{
						lightVisibility = 1.0f;
						shadowRay.dir = -light.dir;
						shadowRay.dir = glm::normalize(shadowRay.dir);
						lightFalloffIntensity = (light.intensity);
						if(glm::dot(normal, shadowRay.dir) > 0)
							inShadow = false;
						break;
					}
				}
				if(!inShadow)
				{
					const Material& material = ray.hitObject->getMaterial();
					if(material.type & Material::DIFFUSE)
					{
						//lightVisibilty: represents how in shadow a point is (1 = completly out of shadow, 0 = completely occluded)
						//dot product gives cosine of angle between the vectors
						//sample() gives the diffuse color of the point
						//dot product must be positive or else the light has no influence
						float dot = glm::dot(normal, shadowRay.dir);
						if(dot > 0.0f)
							finalCol += dot * material.sample(ray, ray.thit0) * material.diffuseCoef;
					}

					if(material.type & Material::BPHONG_SPECULAR)
					{
						//Blinn-Phong shading model specular component
						//specularComponent = specColor * specCoef * lightIntensity * (normal•bisector)ⁿ
						//	ⁿ represents shininess of surface
						glm::vec3 view = glm::normalize(camera.position - shadowRay.pos);
						//bisector ray calculation
						glm::vec3 bisector = glm::normalize(view + shadowRay.dir);
						glm::vec3 r = reflect(-shadowRay.dir, normal);
						float dot = glm::dot(view, r);
						if(dot > 0.0f)
						{
							finalCol += material.specularColor * material.specCoef
								* std::pow(dot, material.shininess);
						}
					}
					if(material.type & Material::REFRACTIVE)
					{
							
						//////////////////REFRACTION////////////////////////
						/*
						****************SNELL'S LAW******************
						*iorOuter*sin(angleToNormalOuter) = iorInner*sin(angleToNormalInner)
						*sin(angleToNormalInner) = (iorOuter * sin(angleToNormalOuter) / iorInner
						*
						********FINDING sin(angleToNormalOuter)******
						*ray.dir X normal = |ray.dir|*|normal| * sin(angleToNormalOuter) * perp
						*	note: ray.dir and normal are normalized
						*ray.dir X normal = sin(angleToNormalOuter) * perp
						*	note: perp is unit vector orthogonal to ray.dir and normal
						*|ray.dir X normal| = sin(angleToNormalOuter) * |perp|
						*|ray.dir X normal| = sin(angleToNormalOuter)
						**********************************************
						*
						************SNELL'S LAW continued*************
						*sin(angleToNormalInner) = (iorOuter / iorInner) * |ray.dir X normal|
						**********************************************
						*
						*Rotate -normal by angleToNormalInner, to get ray inside object
						*Where this ray leaves the object is the starting point of refractionRay
						*refractionRay has the same direction as the ray that hit to object to begin with
						*
						*/

						if(ray.thit1 > -_INFINITY && material.indexOfRefrac >= 1.0f)
						{
							float ior1 = Material::IOR::AIR;
							float ior2 = material.indexOfRefrac;

							Ray innerRay;
							innerRay.pos = shadowRay.pos;
							//move ray outside sphere
							innerRay.pos += -normal * RAY_EPSILON;
							innerRay.dir = refract(ray.dir, normal, ior1, ior2);

							//Calculate reflection and refraction contributions using Fresnel's equation
							float reflectionRatio;
							float transmitRatio;
							float cos1 = glm::dot(-ray.dir, normal);
							float cos2 = glm::dot(innerRay.dir, -normal);

							float parallel = (ior2*cos1 - ior1*cos2) / (ior2*cos1 + ior1*cos2);
							parallel = std::pow(parallel, 2.0f);
							float perpendicular = (ior1*cos2 - ior2*cos1) / (ior1*cos2 + ior2*cos1);
							perpendicular = std::pow(perpendicular, 2.0f);

							//reflectionRatio is average of s and p polarization contributions
							reflectionRatio = 0.5f * (parallel + perpendicular);

							//transmit ratio is leftover after reflection
							transmitRatio = 1.0f - reflectionRatio;

							float t0, t1;
							if(ray.hitObject->getShape()->intersects(innerRay, t0, t1))
							{
								innerRay.thit0 = t0;
								innerRay.thit1 = t1;
								Ray refractionRay;
								refractionRay.pos = innerRay.pos + innerRay.dir * innerRay.thit0;
								glm::vec3 norm = ray.hitObject->getShape()->calcWorldIntersectionNormal(innerRay);
								
								//calculate refraction ray
								refractionRay.dir = refract(innerRay.dir, -norm, ior2, ior1);

								//refraction ray must be in same direction as intersection normal
								if(glm::dot(refractionRay.dir, norm) > 0.0f)
								{
									refractionRay.pos += norm * RAY_EPSILON;
									finalCol += castRay(refractionRay, depth + 1) * transmitRatio;
								}
							}

							Ray reflectionRay;
							reflectionRay.dir = reflect(reflectionRay.pos - ray.pos, normal);
							reflectionRay.pos = shadowRay.pos;
							reflectionRay.pos += normal * RAY_EPSILON;
							finalCol += castRay(reflectionRay, depth + 1) * reflectionRatio;
						}
					}
					//Add visibility (how much in shadow the point is) and falloff coefficients to the final color calculation
					finalCol *= lightVisibility * lightFalloffIntensity;
				}
			}

			if(ray.hitObject->getMaterial().type & Material::MIRROR)
			{			
				//create reflection ray
				Ray reflectionRay;
				reflectionRay.pos = shadowRay.pos;
				glm::vec3 eyeToSurface = reflectionRay.pos - ray.pos;
				reflectionRay.dir = reflect(eyeToSurface, normal);				

				//add bias to avoid self-intersection
				reflectionRay.pos += normal * RAY_EPSILON;

				//TODO: calculate light lost as reflection depth increases
				float thit0, thit1;
				finalCol += castRay(reflectionRay, thit0, thit1, depth + 1) * (1.0f / (float(depth) + 1.0f)) * ray.hitObject->getMaterial().reflectivity;
			}	
		}

		//add ambient light (no point can be completely black to add realism due to indirect lighting effects)
		finalCol += scene->ambientColor * scene->ambientIntensity;
	}
	return finalCol;
}

glm::vec3 Renderer::castRay(Ray& ray, int depth) const
{
	float x, y;
	return castRay(ray, x, y, depth);
}

