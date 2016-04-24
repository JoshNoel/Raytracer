#include "TriObject.h"
#include <fstream>
#include <istream>
#include "glm/glm.hpp"
#include "MathHelper.h"
#include "BoundingBox.h"


TriObject::TriObject()
	: Shape(glm::vec3(0,0,0))
{
}

TriObject::TriObject(glm::vec3 pos)
	: Shape(pos)
{
}

TriObject::~TriObject()
{
}

bool TriObject::checkTris(const std::vector<Triangle*>* tris, Ray& ray, float& thit0, float& thit1) const
{
	//iterate triangles and test if there is an intersection
	bool hit = false;
	std::vector<Triangle*>::const_iterator i;
	for(i = tris->begin(); i != tris->end(); ++i)
	{
		if((*i)->intersects(ray, thit0, thit1))
		{
			
			if(thit0 < ray.thit0)
				ray.hitTri = *i;
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

		//check tris in left or right node for intersection recursively
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

bool TriObject::loadOBJ(const std::string& path)
{
	int x, y;
	x = y = 0;
        tempMatName = "";
        return loadOBJ(path, 0, tempMatName, x, y);
}

bool TriObject::loadOBJ(std::string path, int startLine, std::string& materialName, int& vertexNum, int& uvNum)
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

	int vertexOffset = vertexNum;
	vertexNum = 0;
	int uvOffset = uvNum;
	uvNum = 0;

	//get rid of lines at beginning of file from stream
	//so import only considers data after its start line
	if(startLine != 0)
	{
		for(int i = 0; i < startLine; ++i)
		{
			getline(ifs, line);
		}
	}

	while(getline(ifs, line))
	{
		//stop when stream reaches start of next object's information
		if(line[0] == 'o')
			break;

		//sets the associated material name in the material library
		if(line.find("usemtl") != std::string::npos)
		{
			int spacePos = line.find_first_of(' ');
			materialName = line.substr(spacePos + 1);
		}

		//imports vertices that are a part of the object
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
			points[0] = (vertices.at(std::stoi(line.substr(spacePos + 1, slashPos)) - 1 - vertexOffset));

			spacePos = line.find(' ', spacePos + 1);
			slashPos = line.find('/', slashPos + 1);
			points[1] = (vertices.at(std::stoi(line.substr(spacePos + 1, slashPos)) - 1 - vertexOffset));

			spacePos = line.find(' ', spacePos + 1);
			slashPos = line.find('/', slashPos + 1);
			points[2] = (vertices.at(std::stoi(line.substr(spacePos + 1, slashPos)) - 1 - vertexOffset));

			//import uv coordinate order
			slashPos = line.find_first_of('/');
			spacePos = line.find(' ', slashPos + 1);
			//find vertex at the index given in the file ( vertices stored in vertex array)
			points_uv[0] = (uvCoords.at(std::stoi(line.substr(slashPos + 1, spacePos)) - 1 - uvOffset));

			spacePos = line.find(' ', spacePos + 1);
			slashPos = line.find('/', slashPos + 1);
			points_uv[1] = (uvCoords.at(std::stoi(line.substr(slashPos + 1, spacePos)) - 1 - uvOffset));

			spacePos = line.find(' ', spacePos + 1);
			slashPos = line.find('/', slashPos + 1);
			points_uv[2] = (uvCoords.at(std::stoi(line.substr(slashPos + 1, spacePos)) - 1 - uvOffset));
			//end import uv coordinate order

			//create a bounding box for the imported triangle
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
			tris.back()->setPosition(this->position);
		}
	}

	//add to offset for use in the next object imported by GeometryObj::loadOBJ
	vertexNum = vertexOffset + vertices.size();
	uvNum = uvOffset + uvCoords.size();

	//set up object bounding box
	aabb = tris[0]->aabb;
	for(unsigned int i = 1; i < tris.size(); ++i)
		aabb.join(tris[i]->aabb);
	if(ifs.bad())
		return false;

	ifs.close();
	return true;
}
