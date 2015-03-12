#include "TriObject.h"
#include <fstream>
#include <istream>

TriObject::TriObject(glm::vec3 pos, Material mat)
	: Object(pos, mat), collTriIndex(0)
{
}

TriObject::~TriObject()
{
}

bool TriObject::intersects(const Ray ray, float& t0, float& t1) const
{
	//iterate vertices
	float minT = t0;
	bool hit = false;
	for(unsigned int i = 0; i < this->verts.size(); i += 3)
	{
		glm::vec3 A = (verts[i] + this->position);
		glm::vec3 B = (verts[i + 1] + this->position);
		glm::vec3 C = (verts[i + 2] + this->position);

		glm::vec3 normal = glm::cross(B - A, C - A);
		normal = glm::normalize(normal);
		glm::vec3 position = (A + B + C) / 3.0f;
		
		float y = glm::dot(normal, ray.dir);
		//ray is parrellel to plane of triangle
		if(y < 1e-9)
			continue;
		float x = glm::dot(normal, position - ray.pos);
		float t = (x / y);

		glm::vec3 intersection = ray.pos + ray.dir*t;

		if(glm::dot(glm::cross(B - A, intersection - A), normal) > 0)
		{
			if(glm::dot(glm::cross(C - B, intersection - B), normal) > 0)
			{
				if(glm::dot(glm::cross(A - C, intersection - C), normal) > 0)
				{
					if(t < t0)
					{
						t0 = t;
						hit = true;
					}
				}

			}
		}

	}

	return hit;
}

glm::vec3 TriObject::calcNormal(glm::vec3 p0) const
{
	//set p1, p2, and p3 to triangle vertices in world space
	long index = this->collTriIndex;
	glm::vec3 p1 = this->verts[index] + this->position;
	glm::vec3 p2 = this->verts[index + 1] + this->position;
	//TODO read in normals
	return glm::cross(p0 - p1, p2-p1);
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
			int spacePos = line.find_first_of(' ');
			int nextSpace = line.find(' ', spacePos + 1);
			verts.push_back(vertices.at(std::stoi(line.substr(spacePos+1, nextSpace))-1));

			spacePos = nextSpace;
			nextSpace = line.find(' ', spacePos + 1);
			int j = std::stoi(line.substr(spacePos + 1, nextSpace));
			verts.push_back(vertices.at(std::stoi(line.substr(spacePos+1, nextSpace))-1));

			spacePos = nextSpace;
			nextSpace = line.find('\n', spacePos + 1);
			verts.push_back(vertices.at(std::stoi(line.substr(spacePos+1, nextSpace))-1));
		}
	}
	if(ifs.bad())
		return false;

	aabb.minBounds += position;
	aabb.maxBounds += position;
	ifs.close();
	return true;
}