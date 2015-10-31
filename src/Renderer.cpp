#include "Renderer.h"
#include "MathHelper.h"
#include <iostream>
#include "TriObject.h"
#include "GeometryObj.h"
#include "glm\gtx\rotate_vector.hpp"
#include <iterator>



const float Renderer::SHADOW_RAY_LENGTH = 50.0f;
const int Renderer::NUM_THREADS = 10;

Renderer::Renderer(const Scene* s, Image* i)
	:image(i), camera(), scene(s), distributionX(0, std::nextafterf(1.0f, FLT_MAX)),
	distributionY(0, std::nextafterf(1.0f, FLT_MAX)), mutex()
{
	//initialize rng
	std::random_device device;
	rng.seed(device());
}

Renderer::~Renderer()
{
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
	/*float ratio = ior1 / ior2;
	glm::vec3 dirOnNorm = glm::dot(-dir, norm) * norm;

	glm::vec3 refractedVec = -dirOnNorm + (ratio * (dir + dirOnNorm));

	return glm::normalize(refractedVec);*/
	glm::vec3 dirOnNorm = glm::dot(-dir, norm)*norm;

	//sin(theta1) = |perpToNormal| because hypotonuse (the incident vector) has mag. of 1
	glm::vec3 iPerpToNorm = dir + dirOnNorm;
	float ratio = ior1 / ior2;
	glm::vec3 rPerpToNorm = ratio * iPerpToNorm;
	float thetaInner = std::asinf(glm::length(rPerpToNorm));
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

void Renderer::render()
{
	if(image == nullptr) return;
	if(scene->objectList.size() == 0) return;

	//primary.pos(0, 0, camera.focalLength);
	camera.calculate(image->getAR());

	int imageSegmentLength = std::floorf(image->height / NUM_THREADS);

	//assign threads vertical sections of image to render
	std::array<renderThread, NUM_THREADS> threadArray;
	if(image->height % NUM_THREADS == 0)
	{
		for(int threadNum = 0; threadNum < NUM_THREADS; ++threadNum)
		{
			threadArray[threadNum].start = imageSegmentLength * threadNum;
			threadArray[threadNum].end = (imageSegmentLength * threadNum) + imageSegmentLength;
		}
	}
	else
	{
		for(int threadNum = 0; threadNum < NUM_THREADS; ++threadNum)
		{
			threadArray[threadNum].start = imageSegmentLength * threadNum;
			threadArray[threadNum].end = (imageSegmentLength * threadNum) + imageSegmentLength;
		}
		threadArray[NUM_THREADS - 1].end += image->height % NUM_THREADS;
	}

	//execute threads
	for(int threadNum = 0; threadNum < NUM_THREADS; ++threadNum)
	{
		//initialize thread image data
		int height = threadArray[threadNum].end - threadArray[threadNum].start;
		threadArray[threadNum].data = new glm::vec3[height * image->width];
		//start thread
		threadArray[threadNum].m_thread = std::thread(&Renderer::startThread, this, &threadArray[threadNum]);
	}
	//join threads so that the main loop waits for rendering to finish
	for(int threadNum = 0; threadNum < NUM_THREADS; ++threadNum)
	{
		threadArray[threadNum].m_thread.join();
	}
	//copy image data from each thread to the actual image
	for(int threadNum = 0; threadNum < NUM_THREADS; ++threadNum)
	{
		int height = threadArray[threadNum].end - threadArray[threadNum].start;
		int start = threadArray[threadNum].start;
		std::copy(&threadArray[threadNum].data[0], 
			&threadArray[threadNum].data[height * image->width], 
			&image->data[start * image->width]);
	}

	//traverse image columns
	/*float x, y;
	x = y = 0;
	Ray primary;

	glm::vec3 color;
	for(int i = 0; i < image->height; ++i)
	{

		//traverse rows
		for(int j = 0; j < image->width; ++j)
		{
			glm::vec3 color;
			for(int primarySample = 0; primarySample < scene->PRIMARY_SAMPLES; ++primarySample)
			{
				//normalize x
				x = (2.0f * (j + distributionX(rng)) / image->width) - 1.0f;
				//normalize y
				y = 1.0f - (2.0f * (i + distributionY(rng)) / image->height);

				//x*right in terms of fov == x in terms of right axis
				//primary.dir = glm::vec3(float(x)*image->getAR()*tan(camera.fov/2), float(y)*tan(camera.fov/2), -1) - primary.pos;
				primary.dir = x*camera.right + y*camera.up + camera.direction;
				primary.dir = glm::normalize(primary.dir);

				//Check collision...generate shadow rays
				primary.thit0 = camera.viewDistance;

				float thit0, thit1;
				color += glm::min(castRay(primary, thit0, thit1, 0), glm::vec3(255, 255, 255));
			}
			image->data[i*image->width + j] = color / float(scene->PRIMARY_SAMPLES);
		}
	}*/
			//GeometryObj* hitObject = scene->objectList[primary.hitIndex].get();
				////Create shadow ray at intersection to check if point is in a shadow (there is another object between
				////the point and a light
				//Ray shadowRay;
				//Ray reflectionRay;
				//reflectionRay.pos = shadowRay.pos = primary.pos + primary.dir * primary.thit;
				//glm::vec3 finalCol = glm::vec3(0, 0, 0);

				////iterate lights
				//for(auto light : scene->lightList)
				//{
				//	if(light.intensity > 50.0f)
				//		light.intensity = 50.0f;
				//	if(light.intensity < 0.0f)
				//		light.intensity = 0.0f;
				//	bool inShadow = false;
				//	glm::vec3 normal = hitObject->getShape()->calcWorldIntersectionNormal(shadowRay.pos);
				//	normal = glm::normalize(normal);
				//
				//	float lightFalloffIntensity;
				//	switch(light.type)
				//	{
				//		case Light::POINT:
				//		{
				//			shadowRay.dir = light.pos - shadowRay.pos;
				//			float lightRadius = glm::length(shadowRay.dir);
				//			shadowRay.dir = glm::normalize(shadowRay.dir);
				//			lightFalloffIntensity = (light.intensity) / (4 * _PI_ * lightRadius);
				//			break;
				//		}
				//		case Light::DIRECTIONAL:
				//		{
				//			shadowRay.dir = -light.dir;
				//			shadowRay.dir = glm::normalize(shadowRay.dir);
				//			lightFalloffIntensity = (light.intensity);
				//			break;
				//		}
				//	}

				//	//Create shadow ray from intersection point to light
				//	
				//	//if pointing opposite directions skip light
				//	if(glm::dot(normal, shadowRay.dir) <= 0)
				//		continue;
				//	//check if in shadow
				//	if(light.castsShadow)
				//	{
				//		for(unsigned s2 = 0; s2 < scene->objectList.size(); ++s2)
				//		{
				//			if(s2 == primary.hitIndex) continue;
				//			float temp0, temp1;
				//			temp0 = SHADOW_RAY_LENGTH;
				//			temp1 = -SHADOW_RAY_LENGTH;
				//			if(scene->objectList[s2]->getShape()->intersects(shadowRay, &temp0, &temp1))
				//			{
				//				if(temp0 > 0)
				//					inShadow = true;
				//			}
				//		}
				//	}
				//	if(!inShadow)
				//	{
				//		finalCol += hitObject->getMaterial().color * glm::max(0.0f, glm::dot(normal, shadowRay.dir)) * lightFalloffIntensity;						
				//	}

				//	//check reflection
				//	/******REFLECTION RAY********
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

				//	glm::vec3 eyeToSurface = reflectionRay.pos - primary.pos;
				//	reflectionRay.dir = glm::normalize(eyeToSurface - (2 * glm::dot(eyeToSurface, normal) * normal));

				//	
				//	//TODO Specular material props
				//	for(unsigned s2 = 0; s2 < scene->objectList.size(); ++s2)
				//	{
				//		if(s2 == primary.hitIndex) continue;
				//		float temp0, temp1;
				//		temp0 = SHADOW_RAY_LENGTH;
				//		temp1 = -SHADOW_RAY_LENGTH;
				//		if(scene->objectList[s2]->getShape()->intersects(reflectionRay, &temp0, &temp1))
				//		{
				//			if(temp0 > 0)
				//				finalCol += scene->objectList[s2]->getMaterial().sample(reflectionRay, temp0) * hitObject->getMaterial().specCoef;
				//		}
				//	}
				//}
				//image->data[i*image->width + j] = glm::min(glm::vec3(255, 255, 255),
				//	finalCol + scene->ambientColor*scene->ambientIntensity);
		//}
	//}
}

void Renderer::startThread(renderThread* renderThread) const
{
	int start = renderThread->start;
	int end = renderThread->end;
	int width = image->width;
	int samples = scene->PRIMARY_SAMPLES;
	for(int i = start; i < end; ++i)
	{
		//traverse rows
		for(int j = 0; j < width; ++j)
		{
			glm::vec3 color;

			for(int primarySample = 0; primarySample < samples; ++primarySample)
			{
				float x, y;
				x = y = 0;
				Ray primary;
				//normalize x
				x = (2.0f * (j + distributionX(rng)) / image->width) - 1.0f;
				//normalize y
				y = 1.0f - (2.0f * (i + distributionY(rng)) / image->height);

				//x*right in terms of fov == x in terms of right axis
				//primary.dir = glm::vec3(float(x)*image->getAR()*tan(camera.fov/2), float(y)*tan(camera.fov/2), -1) - primary.pos;
				primary.dir = x*camera.right + y*camera.up + camera.direction;
				primary.dir = glm::normalize(primary.dir);

				//Check collision...generate shadow rays
				primary.thit0 = camera.viewDistance;

				float thit0, thit1;
				color += glm::min(castRay(primary, thit0, thit1, 0), glm::vec3(255, 255, 255));
			}
			renderThread->data[(i-start)*width + j] = color / float(samples);
		}
	}
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
				if(p0 > ray.thit0 || p0 < 0)
					continue;
				else
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
	bool hit = false;
	hit = hitsObject(ray, x, y);
	return (hit && x > 0);
}

//returns sign of arg
//	negative -> return -1
//	positive -> return 1
template <typename T> int sign(T num)
{
	return (num > 0) - (num < 0);
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
			finalCol += scene->ambientColor * scene->ambientIntensity;
			Ray shadowRay;
			shadowRay.pos = ray.pos + ray.dir * ray.thit0;

			glm::vec3 normal = ray.hitObject->getShape()->calcWorldIntersectionNormal(shadowRay.pos);
			normal = glm::normalize(normal);

			//////////////Calculate transmission and reflection contributions using fresnel's equation///////////////
			for(auto light : scene->lightList)
			{
				bool inShadow = true;

				//////Check if in shadow/////
				if(light.intensity < 0.0f)
					light.intensity = 0.0f;

				float lightFalloffIntensity = 0.0f;
				switch(light.type)
				{
					case Light::POINT:
					{
						//Monte Carlo Method
						//Fire rays from point to areaLight, and average results
						if(light.isAreaLight && light.areaShape != nullptr)
						{
							float halfX = light.areaShape->getDimensions().x / 2.0f;
							float halfY = light.areaShape->getDimensions().y / 2.0f;
							std::uniform_real_distribution<float> distributionX(0, std::nextafterf(1.0f, FLT_MAX));
							std::uniform_real_distribution<float> distributionY(0, std::nextafterf(1.0f, FLT_MAX));

							glm::vec3 averageShadowDir;
							float numHits = 0;
							int index = 1;

							//In order to create grid scene->SHADOW_SAMPLES must be a perfect square
							//TODO fast perfect square check
							for(int sampleX = -std::sqrt(scene->SHADOW_SAMPLES) / 2; sampleX < std::sqrtf(scene->SHADOW_SAMPLES) / 2; sampleX++)
							{
								for(int sampleY = -std::sqrt(scene->SHADOW_SAMPLES) / 2; sampleY < std::sqrtf(scene->SHADOW_SAMPLES) / 2; sampleY++)
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
									float gridSquareSideLength = halfX / (std::sqrtf(numGridSquares) / 2.0f);

									//center of grid square
									//	traverse each column
									glm::vec3 basePos = light.areaShape->position +
										(light.areaShape->getU() * ((sampleX * gridSquareSideLength) - (sign(sampleX) * gridSquareSideLength / 2.0f))) +
										(light.areaShape->getV() * ((sampleY * gridSquareSideLength) - (sign(sampleY) * gridSquareSideLength / 2.0f)));
										
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
									//small displacement added along normal to avoid self-intersection
									toLight.pos += normal * RAY_EPSILON

									//Light only contributes if it faces the object								
									if(glm::dot(toLight.dir, light.areaShape->getNormal()))
									{
										//Intersection point must also have a normal facing the light
										//if there is an object between the intersection and point on light, the point is in shadow
										float dot = glm::dot(normal, toLight.dir);
										if(dot > 0 && !hitsObject(toLight))
										{
											inShadow = false;
											numHits++;

											float lightRadius = glm::length(toLight.dir);
											toLight.dir = glm::normalize(toLight.dir);
											averageShadowDir += toLight.dir;
											//lightFalloffIntensity += (light.intensity * dot) / (4.0f * _PI_ * lightRadius);	
											lightFalloffIntensity += (light.intensity) / (std::powf(lightRadius, 1.0f));
										}
									}
								}
							}
							lightFalloffIntensity /= float(scene->SHADOW_SAMPLES);
							shadowRay.dir = averageShadowDir / float(numHits);
						} 
						else 
						{
							shadowRay.dir = light.pos - shadowRay.pos;
							float lightRadius = glm::length(shadowRay.dir);
							shadowRay.dir = glm::normalize(shadowRay.dir);
							lightFalloffIntensity = (light.intensity) / (4 * _PI_ * lightRadius);
						}
						break;
					}
					case Light::DIRECTIONAL:
					{
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
						finalCol += glm::dot(normal, shadowRay.dir) * ray.hitObject->getMaterial().sample(ray, ray.thit0);
					}

					if(material.type & Material::BPHONG_SPECULAR)
					{
						//Blinn-Phong shading model specular component
						//specularComponent = specColor * specCoef * lightIntensity * (normal•bisector)ⁿ
						//	ⁿ represents shininess of surface
						glm::vec3 view = glm::normalize(camera.position - shadowRay.pos);
						//bisector ray calculation
						glm::vec3 bisector = glm::normalize(view + shadowRay.dir);

						finalCol += material.specularColor * material.specCoef
							* std::powf(glm::dot(normal, bisector), material.shininess);
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

						if(ray.thit1 > -_INFINITY && ray.hitObject->getMaterial().indexOfRefrac >= 1.0f)
						{
							float ior1 = Material::IOR::AIR;
							float ior2 = ray.hitObject->getMaterial().indexOfRefrac;

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
							parallel = std::powf(parallel, 2.0f);
							float perpendicular = (ior1*cos2 - ior2*cos1) / (ior1*cos2 + ior2*cos1);
							perpendicular = std::powf(perpendicular, 2.0f);

							//reflectionRatio is average of s and p polarization contributions
							reflectionRatio = 0.5f * (parallel + perpendicular);

							//transmit ratio is leftover after reflection
							transmitRatio = 1.0f - reflectionRatio;

							float t0, t1;
							if(ray.hitObject->getShape()->intersects(innerRay, t0, t1))
							{
								Ray refractionRay;
								refractionRay.pos = innerRay.pos + innerRay.dir * t0;
								glm::vec3 norm = ray.hitObject->getShape()->calcWorldIntersectionNormal(refractionRay.pos);
								
								//calculate refraction ray
								refractionRay.dir = refract(innerRay.dir, -norm, ior2, ior1);

								//refraction ray must be in same direction as intersection normal
								if(glm::dot(refractionRay.dir, -norm) > 0.0f)
								{
									refractionRay.pos += norm * RAY_EPSILON;
									finalCol += castRay(refractionRay, depth + 1) * transmitRatio;
								}
							}

							Ray reflectionRay;
							reflectionRay.dir = reflect(ray.dir, normal);
							reflectionRay.pos = shadowRay.pos;
							reflectionRay.pos += normal * RAY_EPSILON;
							finalCol += castRay(reflectionRay, depth + 1) * reflectionRatio;
						}
					}
					finalCol *= lightFalloffIntensity;
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

				float thit0, thit1;
				finalCol += castRay(reflectionRay, thit0, thit1, depth + 1) * ray.hitObject->getMaterial().reflectivity;
			}	
		}
	}
	return finalCol;
}

glm::vec3 Renderer::castRay(Ray& ray, int depth) const
{
	float x, y;
	return castRay(ray, x, y, depth);
}

