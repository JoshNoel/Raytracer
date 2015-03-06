#include "TriObject.h"
#include <fstream>
#include <istream>

TriObject::TriObject(glm::vec3 pos, Material mat)
	: Object(pos, mat)
{
}

TriObject::~TriObject()
{
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

	ifs.close();
	return true;
}