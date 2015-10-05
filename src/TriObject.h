#pragma once
#include "Shape.h"
#include <string>
#include <vector>
#include "Triangle.h"
#include "Node.h"

class TriObject
	: public Shape
{
public:
	TriObject(glm::vec3);
	~TriObject();

	bool loadOBJ(std::string path);
	void initAccelStruct();


	std::vector<Triangle*> tris;
	Node* root;
	
	mutable Triangle* collisionTri;

	//returns if ray intersects with bounding box 
	bool intersects(Ray& ray, float* thit0, float* thit1) const override;
	glm::vec3 calcWorldIntersectionNormal(glm::vec3) const override;

protected:
	SHAPE_TYPE getType() const override{ return SHAPE_TYPE::TRIANGLE_MESH; };


private:

	bool checkTris(const std::vector<Triangle*>* tris, Ray& ray, float* thit0, float* thit1) const;
	bool checkNode(Node* node, Ray& ray, float* thit0, float* thit1) const;
};