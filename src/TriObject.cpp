#include "TriObject.h"
#include <fstream>
#include <istream>
#include "glm\glm.hpp"

TriObject::TriObject(glm::vec3 pos)
	: Shape(pos), collTriIndex(0)
{
}

TriObject::~TriObject()
{
}

bool TriObject::intersects(Ray& ray, float* thit) const
{
	//iterate vertices
	float t0 = INFINITY;
	bool hit = false;
	for(int i = 0; i < this->tris.size(); ++i)
	{
		if(tris[i].intersects(ray, &t0))
		{
			hit = true;
			collTriIndex = i;
		}
	}

	return hit;
}

glm::vec3 TriObject::calcIntersectionNormal(glm::vec3 v) const
{
	return tris[collTriIndex].calcIntersectionNormal(v);
}

bool TriObject::loadOBJ(std::string path)
{
	aabb.minBounds = glm::vec3(FLT_MAX, FLT_MAX, -FLT_MAX);
	aabb.maxBounds = glm::vec3(-FLT_MAX, -FLT_MAX, FLT_MAX);
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

			//set up aabb
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
				aabb.maxBounds.z = z;
		}

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
			tris.push_back(Triangle(points, this));
		}
	}
	if(ifs.bad())
		return false;

	aabb.minBounds += position;
	aabb.maxBounds += position;
	ifs.close();
	return true;
}