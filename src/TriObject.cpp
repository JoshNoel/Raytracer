#include "TriObject.h"
#include <fstream>
#include <istream>
#include "glm\glm.hpp"
#include "MathHelper.h"
#include "BoundingBox.h"

bool TriObject::checkTris(const std::vector<Triangle*>* tris, Ray& ray, float& thit0, float& thit1) const
{
	//iterate vertices
	bool hit = false;
	for(unsigned int i = 0; i < tris->size(); ++i)
	{
		if((*tris)[i]->intersects(ray, thit0, thit1))
		{
			if(thit0 < ray.thit0)
				ray.hitTri = (*tris)[i];
			hit = true;
		}	
	}

	return hit;
}

bool TriObject::checkNode(Node* node, Ray& ray, float& thit0, float& thit1) const
{
	if(node == nullptr)
		return false;
	if(node->aabb.intersects(ray))
	{
		//if leaf just check tris in this node
		if(node->isLeaf())
		{
			return checkTris(node->tris, ray, thit0, thit1);
		}

		//check tris in left or right node for intersection
		else
		{
			assert(node->left && node->right);
			bool hit = false;
			if(checkNode(node->right, ray, thit0, thit1))
				hit = true;
			if(checkNode(node->left, ray, thit0, thit1))
				hit = true;

			return hit;
		}
	}
	return false;
}


TriObject::TriObject(glm::vec3 pos)
	: Shape(pos)
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
	return checkNode(root, ray, thit0, thit1);
}

glm::vec3 TriObject::calcWorldIntersectionNormal(const Ray& ray) const
{
	glm::vec3 norm = ray.hitTri->calcWorldIntersectionNormal(ray);
	return invertNormals ? -norm : norm;
}

bool TriObject::loadOBJ(std::string path)
{
	bool hasUV = false;

	int extStart = path.find_last_of('.');
	if(path.substr(extStart) != ".obj")
		return false;

	std::string line;

	std::ifstream ifs;
	ifs.open(path);
	if(!ifs.is_open())
		return false;

	std::vector<glm::vec3> vertices;
	std::vector<glm::vec2> uvCoords;
	while(getline(ifs, line))
	{
		if(line[0] == 'v' && line[1] != 'n' && line[1] != 't')
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
		}

		//import uv coordinates if they exist
		else if(line[0] == 'v' && line[1] == 't')
		{
			hasUV = true;
			int spacePos = line.find_first_of(' ');
			int nextSpace = line.find(' ', spacePos + 1);
			float u = std::stof(line.substr(spacePos + 1, nextSpace));

			spacePos = nextSpace;
			nextSpace = line.find(' ', spacePos + 1);
			float v = std::stof(line.substr(spacePos + 1, nextSpace));

			uvCoords.push_back(glm::vec2(u, v));
		}

		//Create faces from vertex array
		//	format of f line is:
		//	vertexIndex/uv
		else if(line[0] == 'f')
		{
			std::array<glm::vec3, 3> points;
			std::array<glm::vec2, 3> points_uv;
			
			int spacePos = line.find_first_of(' ');
			int slashPos = line.find('/', spacePos + 1);
			//find vertex at the index given in the file ( vertices stored in vertex array)
			points[0] = (vertices.at(std::stoi(line.substr(spacePos+1, slashPos))-1));

			spacePos = line.find(' ', spacePos + 1);
			slashPos = line.find('/', slashPos + 1);
			points[1] = (vertices.at(std::stoi(line.substr(spacePos+1, slashPos))-1));

			spacePos = line.find(' ', spacePos + 1);
			slashPos = line.find('/', slashPos + 1);
			points[2] = (vertices.at(std::stoi(line.substr(spacePos+1, slashPos))-1));

			//import uv coordinate order
			slashPos = line.find_first_of('/');
			spacePos = line.find(' ', slashPos + 1);
			//find vertex at the index given in the file ( vertices stored in vertex array)
			points_uv[0] = (uvCoords.at(std::stoi(line.substr(slashPos + 1, spacePos)) - 1));

			spacePos = line.find(' ', spacePos + 1);
			slashPos = line.find('/', slashPos + 1);
			points_uv[1] = (uvCoords.at(std::stoi(line.substr(slashPos + 1, spacePos)) - 1));

			spacePos = line.find(' ', spacePos + 1);
			slashPos = line.find('/', slashPos + 1);
			points_uv[2] = (uvCoords.at(std::stoi(line.substr(slashPos + 1, spacePos)) - 1));
			//end import uv coordinate order

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

			bbox.minBounds += position;
			bbox.maxBounds += position;

			tris.push_back(new Triangle(points, true));
			tris.back()->setUVCoords(points_uv);
			tris.back()->aabb = bbox;
			tris.back()->parent = this->parent;
			tris.back()->position = this->position;
		}
	}

	//set up object bounding box
	aabb = tris[0]->aabb;
	for(unsigned int i = 1; i < tris.size(); ++i)
		aabb.join(tris[i]->aabb);
	if(ifs.bad())
		return false;

	ifs.close();
	return true;
}