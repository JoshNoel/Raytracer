#pragma once
#include "Shape.h"
#include "CudaDef.h"
#include "glm/glm.hpp"


class Sphere
	: public Shape
{
	/*
	*****************SPHERE IMPLICIT INTERSECTION***********************
	*pos = point position, C = sphere center, r = radius
	*Eq.1 sphere definition (algebraic): (x - C.x)² + (y - C.y)² + (z - C.z)² = r²
	*E1.2 sphere definition (vector): ||pos - C||² = r²
	*Eq.3:	||pos - C||² - r² = 0
	*If ray intersects the sphere a position on the ray with be a solution to Eq.1, where C and r are constants
	*o = ray origin, t = parametric time, d = ray direction
	*Eq.4 Position on ray: posOnRay = o + dt
	*Eq.5 Substitute Eq.4 into Eq.3: ||o + dt - C||² - r² = 0 <-> (o - C + dt)•(o - C + dt) - r² = 0
	*	note: ||a||² = a•a
	*Eq.6 expand dot product using distributive rulse: (o-C)•(o-C) + 2((o-C)•dt) + d²t² - r² = 0
	*	note: d is normalized, so d•d = 1
	*	note: dot product is distributive, so one can multiply the binomial
	*	(a+b)•(a+b) = a•a + 2(a•b) + b•b
	*	a = o-C, b = dt
	*Eq.7 Simplified Eq.6: t² + 2((o-C)•d)*t + (||o-c||² - r²)= 0
	*These relationships are true for any point on a given ray and sphere
	*One can solve for t values using quadratic equation with constants:
	*	a = 1
	*	b = 2((o-C)•d)
	*	c = ||o-c||² - r²
	*Find determinent to see if intersection exists
	*If 1 exists ray is tangent
	*If 2 exist the smaller t value(t₀) is the first intersection, the larger is the exit(t₁)
	*---------(-----)--------->
	*		  t₀	t₁
	*/


public:
	CUDA_DEVICE CUDA_HOST Sphere(glm::vec3 pos = glm::vec3(0,0,0), float radius = 1.0f);
	CUDA_DEVICE CUDA_HOST ~Sphere();

	struct parameters
		: public Shape::parameters
	{
		parameters(const glm::vec3& position, float radius)
		{
			data = new Data();
			data->position = position;
			data->radius = radius;
			num_params = 2;
		}
	
		explicit parameters(const glm::vec3& position)
		{
			data = new Data();
			data->position = position;
			num_params = 1;
		}

		parameters()
		{
			data = new  Data();
			num_params = 0;
		}

		parameters(parameters&& params)
		{
			data = new Data();
			data->position = params.getPosition();
			data->radius = params.getRadius();
			num_params = params.num_params;
		}

		glm::vec3 getPosition() const{ return data->position;  }
		float getRadius() const { return data->radius;  }

		size_t getParamSize(int num_params) override
		{
			assert(num_params < MAX_PARAMS);
			return PARAM_SIZES[num_params];
		}

		size_t getParamSize() override
		{
			return PARAM_SIZES[num_params];
		}

		void* getParam(unsigned i) override
		{
			switch (i)
			{
			default:
				assert(i < num_params);
			case 0:
				return &data->position;
			case 1:
				return &data->radius;
			}
		}

	private:
		static const int MAX_PARAMS = 3;
		//holds additive size of params
		static const int PARAM_SIZES[MAX_PARAMS];

		struct Data : public Managed
		{
			glm::vec3 position = glm::vec3(0, 0, 0);
			float radius = 1.0f;
		};
		Data* data;
	};

	CUDA_DEVICE CUDA_HOST SHAPE_TYPE getType() const override { return SHAPE_TYPE::SPHERE; }

	/*Tests for ray sphere intersection using Math.solveQuadratic(see Sphere.h)
	* returns result of determinant
	*	-1 = no intersection
	*	 0 = ray is tangent
	*	 1 = 2 intersections
	*
	* t0 will contain position of first intersection(or only if tangent)
	* t1 will contain position of second intersection
	*/
	CUDA_DEVICE bool intersects(Ray& ray, float& thit0, float& thit1) const override;
	CUDA_DEVICE glm::vec3 calcWorldIntersectionNormal(const Ray& ray) const override;

	float radius;
};

