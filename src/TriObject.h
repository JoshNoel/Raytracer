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

	TriObject();
	TriObject(glm::vec3);
	~TriObject();

	bool loadOBJ(const std::string& path);
	//startLine: start of object data in .obj file
	//vertexNum: set to number to offset vertice index by in 'f' lines because obj does not reset vertex indices by object
	//	on return set to number of vertices in object in order to update offset for next objects
	//uvNum: set to number to offset uv index by in 'f' lines because obj does not reset uv indices by object
	//	on return set to number of uv coordinates in object in order to update offset for next objects
	bool loadOBJ(std::string path, int startLine, std::string& materialName, int& vertexNum, int& uvNum);
	void initAccelStruct();


	std::vector<Triangle*> tris;
	Node* root;
	
	//returns if ray intersects with bounding box 
	bool intersects(Ray& ray, float& thit0, float& thit1) const override;
	glm::vec3 calcWorldIntersectionNormal(const Ray& ray) const override;

	void flipNormals(bool flip)
	{
		this->invertNormals = flip;
	}


protected:
	SHAPE_TYPE getType() const override{ return SHAPE_TYPE::TRIANGLE_MESH; };


private:

	bool checkTris(const std::vector<Triangle*>* tris, Ray& ray, float& thit0, float& thit1) const;
	bool checkNode(Node* node, Ray& ray, float& thit0, float& thit1) const;

	bool invertNormals;
};