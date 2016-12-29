#pragma once
#include "Shape.h"
#include "MathHelper.h"
#include "managed.h"

//small depth added in order to create an axis aligned bounding box
#ifndef PLANE_DEPTH
	#define PLANE_DEPTH 0.01f
#endif

/*
*(P-P₀)·n=0
*P = input
*P₀ = point on plane (center in this case)
*n = normal vector of plane
*
*/
class Plane :
	public Shape
{
public:
	//position: position of plane
	//xAngle, yAngle, zAngle: angles to rotate normal vector by in radians
	//	default normal is (0,1,0)
	explicit CUDA_DEVICE CUDA_HOST Plane(const glm::vec3& position = glm::vec3(0,0,0), float xAngle = 0.0f, float yAngle = 0.0f, float zAngle = 0.0f, const glm::vec2& dimensions = glm::vec2(1000.0f, 1000.0f));
	CUDA_DEVICE CUDA_HOST ~Plane();

	//need struct of constructor parameters for cuda support
	struct parameters
		: public Shape::parameters
	{
		parameters(const glm::vec3& position, float x_angle, float y_angle, float z_angle, const glm::vec2& dimensions)
		{
			data = new Data();
			data->position = position;
			data->xAngle = x_angle;
			data->yAngle = y_angle;
			data->zAngle = z_angle;
			data->dimensions = dimensions;
		
			num_params = 5;
		}

		parameters(const glm::vec3& position, float x_angle, float y_angle, float z_angle)
		{
			data = new Data();
			data->position = position;
			data->xAngle = x_angle;
			data->yAngle = y_angle;
			data->zAngle = z_angle;
			num_params = 4;
		}

		parameters(const glm::vec3& position, float x_angle, float y_angle)
		{
			data = new Data();
			data->position = position;
			data->xAngle = x_angle;
			data->yAngle = y_angle;
			num_params = 3;
		}

		parameters(const glm::vec3& position, float x_angle)
		{
			data = new Data();
			data->position = position;
			data->xAngle = x_angle;

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
			data = new Data();
			num_params = 0;
		}

		parameters(parameters&& params)
		{
			data = new Data();
			data->position = params.getPosition();
			data->xAngle = params.getXAngle();
			data->yAngle = params.getYAngle();
			data->zAngle = params.getXAngle();
			data->dimensions = params.getDimensions();
			num_params = params.num_params;
		}

		glm::vec3 getPosition() const { return data->position; }

		float getXAngle() const { return data->xAngle; }

		float getYAngle() const { return data->yAngle;	}

		float getZAngle() const { return data->zAngle;	}

		glm::vec2 getDimensions() const { return data->dimensions;	}

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
				return &data->xAngle;
			case 2:
				return &data->yAngle;
			case 3:
				return &data->zAngle;
			case 4:
				return &data->dimensions;
			}
		}

	private:
		static const int MAX_PARAMS = 6;
		//holds additive size of params
		static const int PARAM_SIZES[MAX_PARAMS];

		struct Data : public Managed
		{
			glm::vec3 position = glm::vec3(0, 0, 0);
			float xAngle = 0.0f;
			float yAngle = 0.0f;
			float zAngle = 0.0f;
			glm::vec2 dimensions = glm::vec2(1000.0f, 1000.0f);
		};
		Data* data;
	};

	CUDA_DEVICE CUDA_HOST SHAPE_TYPE getType() const override{ return SHAPE_TYPE::PLANE; }
	CUDA_DEVICE bool intersects(Ray& ray, float& thit0, float& thit1) const override;
	CUDA_DEVICE glm::vec3 calcWorldIntersectionNormal(const Ray& ray) const override;

	CUDA_DEVICE CUDA_HOST void setProperties(float xAngle, float yAngle, float zAngle, glm::vec2 dimensions);
	CUDA_DEVICE CUDA_HOST glm::vec2 getDimensions() const;
	CUDA_DEVICE CUDA_HOST glm::vec3 getNormal() const { return normal; }
	CUDA_DEVICE CUDA_HOST glm::vec3 getU() const { return uVec; }
	CUDA_DEVICE CUDA_HOST glm::vec3 getV() const { return vVec; }

private:
	//x = width
	//	distance along uVec
	//y = length
	//	distance along vVec
	glm::vec2 dimensions;
	glm::vec3 normal;

	//represents normalized "x-axis", of plane's coordinate system
	glm::vec3 uVec;
	//represents normalized "y-axis", of plane's coordinate system
	glm::vec3 vVec;
};

