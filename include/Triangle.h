#pragma once
#include "helper/array.h"
#include "Shape.h"

class TriObject;

//Holds 3 points that create triangle
/*************PLANE INTERSECTION*****************
*For any point in a plane P, where P₀ is position of plane, and N is plane's normal:
*Eq.1: (P-P₀) • N = 0
*	This is true because if point is on plane a vector from the plane to this point is
*	orthogonal with the plane's normal
*Eq.2 substitute ray equation into Eq.1: (o + dt - P₀) • N = 0
*Eq.3 distribute dot product in Eq.2: (dt • N) + ((o-P₀) • N) = 0
*E1.4 solve Eq.2 for t: -[((o-P₀) • N)]/[d • N] = t
**************************************************
*************TRIANGLE INTERSECTION****************
*If the ray intersects the intersection point within the triangle formed by the three points
*then cross product between side vector pointing CCW between points, and vector from start of a side
*to intersection will always point in same direction as the normal
*/
class Triangle : public Shape
{
public:
	template<bool B>
	CUDA_HOST CUDA_DEVICE Triangle(const helper::array<glm::vec3, 3, B>& p, bool calcNormal)
		: Shape(glm::vec3(0, 0, 0))
	{
		points = new helper::array<glm::vec3, 3, true>();
		points_world = new helper::array<glm::vec3, 3, true>();
		uvCoords = new helper::array<glm::vec2, 3, true>();

		*points = p;
		if (calcNormal)
			normal = calcObjectNormal();
		else
			normal = glm::vec3(0, 1, 0);

		position = calcCenter();
	}
	
	template<bool B>
	CUDA_DEVICE CUDA_HOST Triangle(const helper::array<glm::vec3, 3, B>& p, bool calcNormal, glm::vec3 position, const helper::array<glm::vec2, 3>& uv_array, BoundingBox* aabb)
		: Shape(glm::vec3(0, 0, 0))
	{
		points_world = new helper::array<glm::vec3, 3, true>();
		uvCoords = new helper::array<glm::vec2, 3, true>();
		points = new helper::array<glm::vec3, 3, true>();

		*points = p;
		if (calcNormal)
			normal = calcObjectNormal();
		else
			normal = glm::vec3(0, 1, 0);

		this->aabb = aabb;
		setUVCoords(uv_array);
		setPosition(position);
	}

	
	
	CUDA_DEVICE CUDA_HOST ~Triangle()
	{
		delete points;
		delete points_world;
		delete uvCoords;
	}

	struct parameters
		: public Shape::parameters
	{
		template<bool B>
		parameters(const helper::array<glm::vec3, 3, B>& points, bool calc_normal, const glm::vec3& position, const helper::array<glm::vec2, 3, true>& uv_array, BoundingBox* aabb)
		{
			data = new Data();
			data->points = points;
			data->calcNormal = calc_normal;
			data->position = position;
			data->uv_array = uv_array;
			data->aabb = aabb;
			num_params = 5;
		}

		template<bool B>
		parameters(const helper::array<glm::vec3, 3, B>& points, bool calc_normal)
		{
			data = new Data();
			data->points = points;
			data->calcNormal = calc_normal;
		
			helper::array<glm::vec2, 3> a;
			a[0] = glm::vec2();
			a[1] = glm::vec2();
			a[2] = glm::vec2();
			data->uv_array = a;
			data->aabb = new BoundingBox();
			num_params = 2;
		}
		
		parameters(parameters&& params)
		{
			data = new Data();
			data->points = params.getPoints();
			data->calcNormal = params.getCalcNormal();
			data->position = params.getPosition();
			data->uv_array = params.getUvArray();
			data->aabb = new BoundingBox(*params.getBoundingBox());
			num_params = params.num_params;
		}


		helper::array<glm::vec3, 3, true> getPoints() const
		{ return data->points; }

		bool getCalcNormal() const { return data->calcNormal; }

		glm::vec3 getPosition() const { return data->position; }

		helper::array<glm::vec2, 3, true> getUvArray() const { return data->uv_array; }

		BoundingBox* getBoundingBox() const { return data->aabb; }

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
				return &data->points;
			case 1:
				return &data->calcNormal;
			case 2:
				return &data->position;
			case 3:
				return &data->uv_array;
			case 4:
				return data->aabb;
			}
		}

	private:
		static const int MAX_PARAMS = 6;
		//holds additive size of params
		static const int PARAM_SIZES[MAX_PARAMS];

		struct Data : public Managed
		{
			helper::array<glm::vec3, 3, true> points;
			bool calcNormal = false;
			glm::vec3 position = glm::vec3(0, 0, 0);
			helper::array<glm::vec2, 3, true> uv_array;
			BoundingBox* aabb;
		};
		Data* data;
	};

	CUDA_DEVICE CUDA_HOST SHAPE_TYPE getType() const override { return SHAPE_TYPE::TRIANGLE; }

	//add uv coordinates to the triangle
	//	coordinate order matches with vertex order in points array
	CUDA_DEVICE CUDA_HOST void setUVCoords(const helper::array<glm::vec2, 3>&);

	//returns the precaclulated normal value
	CUDA_DEVICE glm::vec3 calcWorldIntersectionNormal(const Ray&) const override;

	//returns points in world space
	CUDA_DEVICE CUDA_HOST helper::array<glm::vec3, 3, true> getWorldCoords() const;

	//returns points in object space
	CUDA_DEVICE CUDA_HOST helper::array<glm::vec3, 3, true> getPoints() const;

	//returns uvCoordinates
	CUDA_DEVICE CUDA_HOST bool getUV(helper::array<glm::vec2, 3, true>& coords) const;

	CUDA_DEVICE bool intersects(Ray& ray, float& thit0, float& thit1) const override;

	
	
	glm::vec3 normal;

	bool hasUV = false;

	//updates position, points_world, and triangleCenter
	CUDA_DEVICE CUDA_HOST void setPosition(const glm::vec3& pos) override;

private:
	//used to compute normal at creation
	CUDA_DEVICE CUDA_HOST glm::vec3 calcObjectNormal() const;

	helper::array<glm::vec3, 3, true>* points;
	helper::array<glm::vec3, 3, true>* points_world;
	helper::array<glm::vec2, 3, true>* uvCoords;

	glm::vec3 triangleCenter;

	//calculate center of triangle
	CUDA_DEVICE CUDA_HOST glm::vec3 calcCenter() const;
};


