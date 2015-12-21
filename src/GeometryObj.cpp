#include "GeometryObj.h"
#include <iostream>
#include <fstream>
#include "TriObject.h"

GeometryObj::GeometryObj(std::shared_ptr<Shape> s, const Material& mat)
	: material(mat), shape(s)
{
	s->parent = this;
}

GeometryObj::GeometryObj(std::shared_ptr<Shape> s, const Material& mat, const std::string& name)
	: GeometryObj(s, mat)
{
	this->name = name;
}


GeometryObj::~GeometryObj()
{
}

bool GeometryObj::loadOBJ(const std::string& path, std::vector<std::unique_ptr<GeometryObj>>* objectList, const glm::vec3& position, bool flipNormals)
{
	bool hasUV = false;

	int extStart = path.find_last_of('.');
	if(path.substr(extStart) != ".obj")
		return false;
	
	std::string materialLibPath;
	std::string line;

	std::ifstream ifs;
	ifs.open(path);
	if(!ifs.is_open())
		return false;

	int objectLineCounter = 0;
	int vertexOffset = 0;
	int uvOffset = 0;

	while(getline(ifs, line))
	{
		objectLineCounter++;

		if(line.find("mtllib") != std::string::npos)
		{
			int spacePos = line.find_first_of(' ');
			materialLibPath = line.substr(spacePos + 1);
		}
		if(line[0] == 'o')
		{
			//initialize object name
			int spacePos = line.find_first_of(' ');
			std::string objectName = line.substr(spacePos + 1);

			//initialize triangle object
			std::shared_ptr<TriObject> triObject = std::make_shared<TriObject>(position);
			std::string materialName;
			triObject->loadOBJ(path, objectLineCounter, materialName, vertexOffset, uvOffset);
			triObject->initAccelStruct();
			triObject->flipNormals(flipNormals);

			//initialize material
			Material material;
			//generate mtl file path (in same directory as .obj, but named $materialLibPath
			int lastSlash = path.find_last_of("/");
			std::string mtlPath = path.substr(0, lastSlash + 1) + materialLibPath;
			material.loadMTL(mtlPath, materialName);

			objectList->push_back(std::make_unique<GeometryObj>(triObject, material, objectName));
		}
	}

	if(ifs.bad())
		return false;

	ifs.close();
	return true;
}
