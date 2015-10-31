#include "TriObject.h"
#include <fstream>
#include <istream>
#include "glm\glm.hpp"
#include "MathHelper.h"
#include "BoundingBox.h"

bool TriObject::checkTris(const std::vector<Triangle*>* tris, Ray& ray, float& thit0, float& thit1) const
{
	//iterate vertices
	float t0 = _INFINITY;
	float t1 = -_INFINITY;
	bool hit = false;
	for(int i = 0; i < tris->size(); ++i)
	{
		if((*tris)[i]->intersects(ray, t0, t1))
		{
			if(t0 < thit0)
			{
				thit0 = t0;
				collisionTri = (*tris)[i];
				hit = true;
			}
			if(t1 > thit1) thit1 = t1;
		}
	}
	if(thit0 > thit1)
		std::swap(thit0, thit1);

	return hit;
}

bool TriObject::checkNode(Node* node, Ray& ray, float& thit0, float& thit1) const
{
	if(node == nullptr)
		return false;
	if(node->aabb.intersects(ray))
	{
		if(node->isLeaf())
		{
			return checkTris(node->tris, ray, thit0, thit1);
		}

		if(node->left != nullptr)
		{
			if(checkNode(node->left, ray, thit0, thit1))
				return true;
		}

		if(node->right != nullptr)
		{
			if(checkNode(node->right, ray, thit0, thit1))
				return true;
		}
	}
	return false;
}


TriObject::TriObject(glm::vec3 pos)
	: Shape(pos), collisionTri(nullptr)
{
}

TriObject::~TriObject()
{
}

void TriObject::initAccelStruct()
{
	root = new Node();
	root->createNode(&tris, 0);
}

bool TriObject::intersects(Ray& ray, float& thit0, float& thit1) const
{
	//iterate vertices
	float t0 = _INFINITY;
	float t1 = -_INFINITY;
	bool hit = false;
	Node* node = root;
	hit = checkNode(node, ray, t0, t1);
	if(t0 > t1)
		std::swap(t0, t1);
	if(t0 < thit0)
		thit0 = t0;
	if(t1 > thit1)
		thit1 = t1;
	return hit;
}

glm::vec3 TriObject::calcWorldIntersectionNormal(glm::vec3 v) const
{
	return collisionTri->calcWorldIntersectionNormal(v);
}

bool TriObject::loadOBJ(std::string path)
{
	int extStart = path.find_last_of('.');
	if(path.substr(extStart) != ".obj")
		return false;

	std::string line;

	std::ifstream ifs;
	ifs.open(path);
	if(!ifs.is_open())
		return false;

	std::vector<glm::vec3> vertices;
	while(getline(ifs, line))
	{
		if(line[0] == 'v')
		{
			int spacePos = line.find_first_of(' ');
			int nextSpace = line.find(' ', spacePos + 1);
			float x = std::stof(line.substr(spacePos+1, nextSpace));
			
			spacePos = nextSpace;
			nextSpace = line.find(' ', spacePos + 1);
			float y = std::stof(line.substr(spacePos+1, nextSpace));

			spacePos = nextSpace;
			nextSpace = line.find('\n', spacePos + 1);
			float z = std::stof(line.substr(spacePos+1, nextSpace));

			vertices.push_back(glm::vec3(x, y, z));

			/*//set up aabb
			if(x < aabb.minBounds.x)
				aabb.minBounds.x = x;
			if(y < aabb.minBounds.y)
				aabb.minBounds.y = y;
			if(z > aabb.minBounds.z)
				aabb.minBounds.z = z;

			if(x > aabb.maxBounds.x)
				aabb.maxBounds.x = x;
			if(y > aabb.maxBounds.y)
				aabb.maxBounds.y = y;
			if(z < aabb.maxBounds.z)
				aabb.maxBounds.z = z;*/
		}

		//Create faces from vertex array
		if(line[0] == 'f')
		{
			std::array<glm::vec3, 3> points;

			int spacePos = line.find_first_of(' ');
			int nextSpace = line.find(' ', spacePos + 1);
			points[0] = (vertices.at(std::stoi(line.substr(spacePos+1, nextSpace))-1));

			spacePos = nextSpace;
			nextSpace = line.find(' ', spacePos + 1);
			int j = std::stoi(line.substr(spacePos + 1, nextSpace));
			points[1] = (vertices.at(std::stoi(line.substr(spacePos+1, nextSpace))-1));

			spacePos = nextSpace;
			nextSpace = line.find('\n', spacePos + 1);
			points[2] = (vertices.at(std::stoi(line.substr(spacePos+1, nextSpace))-1));

			BoundingBox bbox;
			bbox.minBounds.x = bbox.maxBounds.x = points[0].x;
			bbox.minBounds.y = bbox.maxBounds.y = points[0].y;
			bbox.minBounds.z = bbox.maxBounds.z = points[0].z;

			for(int i = 1; i < 3; ++i)
			{
				if(points[i].x < bbox.minBounds.x)
					bbox.minBounds.x = points[i].x;
				if(points[i].x > bbox.maxBounds.x)
					bbox.maxBounds.x = points[i].x;

				if(points[i].y < bbox.minBounds.y)
					bbox.minBounds.y = points[i].y;
				if(points[i].y > bbox.maxBounds.y)
					bbox.maxBounds.y = points[i].y;

				//-z axis goes into scene, so the greater the z value the closer it is to the camera
				//therefore bbox.minBounds.z > bbox.maxBounds.z
				if(points[i].z > bbox.minBounds.z)
					bbox.minBounds.z = points[i].z;
				if(points[i].z < bbox.maxBounds.z)
					bbox.maxBounds.z = points[i].z;
			}
			/*for(int i = 0; i < 3; ++i)
			{
				if(points[i].x < bbox.minBounds.x)
					bbox.minBounds.x = points[i].x;
				if(points[i].y < bbox.minBounds.y)
					bbox.minBounds.y = points[i].y;
				//maxBounds.z < minBounds.z because -z away from camera direction
				if(points[i].z > bbox.minBounds.z)
					bbox.minBounds.z = points[i].z;

				if(points[i].x > bbox.maxBounds.x)
					bbox.maxBounds.x = points[i].x;
				if(points[i].y > bbox.maxBounds.y)
					bbox.maxBounds.y = points[i].y;
				//maxBounds.z < minBounds.z because -z away from camera direction
				if(points[i].z < bbox.maxBounds.z)
					bbox.maxBounds.z = points[i].z;
			}*/

			bbox.minBounds += position;
			bbox.maxBounds += position;

			tris.push_back(new Triangle(points));
			tris.back()->aabb = bbox;
			tris.back()->parent = this->parent;
			tris.back()->position = this->position;
		}
	}

	//set up object bounding box
	aabb = tris[0]->aabb;
	for(int i = 1; i < tris.size(); ++i)
		aabb.join(tris[i]->aabb);
	if(ifs.bad())
		return false;

	ifs.close();
	return true;
}