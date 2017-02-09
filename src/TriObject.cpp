#include "TriObject.h"
#include "CudaDef.h"
#include <fstream>
#include "glm/glm.hpp"
#include "BoundingBox.h"
#include "CudaLoader.h"

const int TriObject::parameters::PARAM_SIZES[TriObject::parameters::MAX_PARAMS] = { 
0,
sizeof(glm::vec3),
sizeof(glm::vec3) + sizeof(bool) };

CUDA_DEVICE TriObject::TriObject(glm::vec3 pos, bool flipNormals)
	: Shape(pos), root(nullptr), invertNormals(flipNormals)
{
#ifdef USE_CUDA
	gpuData = new GpuData();
#endif
}

CUDA_DEVICE TriObject::~TriObject()
{
#ifndef USE_CUDA
	for(unsigned i = 0; i < tris.size(); i++) {
		delete tris[i];
	}
#endif
	delete root;
}

CUDA_DEVICE bool TriObject::checkTris(Node* node, Ray& ray, float& thit0, float& thit1) const
{
	//iterate triangles and test if there is an intersection
	bool hit = false;
	//use gpuData if using cuda
	for (unsigned i = 0; i <node->m_data->trisSize; i++)
	{
		if (node->m_data->tris[i]->intersects(ray, thit0, thit1))
		{

			if (thit0 < ray.thit0)
				ray.hitTri = node->m_data->tris[i];
			hit = true;
		}
	}
	return hit;
}

CUDA_DEVICE bool TriObject::checkNode(Node* node, Ray& ray, float& thit0, float& thit1) const
{
	if(node == nullptr)
		return false;
	if(node->aabb->intersects(ray))
	{
		//if leaf just check tris in this node
		if(node->isLeaf())
		{
			return checkTris(node, ray, thit0, thit1);
		}else{
			//check tris in left or right node for intersection recursively

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

CUDA_DEVICE void TriObject::initAccelStruct()
{
	root = new Node();
#ifdef USE_CUDA
	//TODO: create a device function that recursivly creates kernels
	root->createNode(gpuData->tris, gpuData->trisSize, 0);
#else
	root->createNode(tris, 0);
#endif
}

CUDA_DEVICE bool TriObject::intersects(Ray& ray, float& thit0, float& thit1) const
{
	return checkNode(root, ray, thit0, thit1);
}

CUDA_DEVICE glm::vec3 TriObject::calcWorldIntersectionNormal(const Ray& ray) const
{
	glm::vec3 norm = ray.hitTri->calcWorldIntersectionNormal(ray);
	return invertNormals ? -norm : norm;
}

bool TriObject::loadOBJ(std::vector<Triangle**>& tris, BoundingBox* aabb, glm::vec3 position, const std::string& path, CudaLoader& loader)
{
	int x, y;
	x = y = 0;
    std::string tempMatName = "";
    return loadOBJ(tris, aabb, position, path, 0, tempMatName, x, y, loader);
}
//TODO: make object loading multithreaded. May be better use of time to implement fbx support
bool TriObject::loadOBJ(std::vector<Triangle**>& tris, BoundingBox* aabb, glm::vec3 position, std::string path, int startLine, std::string& materialName, int& vertexNum, int& uvNum, CudaLoader& loader)
{
	bool hasUV = false;

	int extStart = path.find_last_of('.');
	if(path.substr(extStart) != ".obj")
		return false;

	//first count number of tris to preallocate tris vector
	std::ifstream face_count_ifs;
	std::string face_count_line;
	long face_count = 0;
	face_count_ifs.open(path);
	if (!face_count_ifs.is_open())
		return false;
	while (getline(face_count_ifs, face_count_line)) {
		if (face_count_line[0] = 'f')
			face_count++;
	}
	face_count_ifs.close();
	tris.reserve(face_count);


	std::string line;
	
	std::ifstream ifs;
	ifs.open(path);
	if(!ifs.is_open())
		return false;

	vector<glm::vec3> vertices;
	vector<glm::vec2> uvCoords;

	bool firstTri = true;

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
			helper::array<glm::vec3, 3, true> points;
			helper::array<glm::vec2, 3, true> points_uv;
			
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

			if (hasUV)
			{
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
			}

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

			//if not using cuda it will fill tris with Triangle* allocated on host
			Triangle** tri = loader.loadShape<Triangle>(points, true, position, points_uv, new BoundingBox(bbox));
			tris.push_back(tri);
			if(firstTri)
			{
				*aabb = bbox;
				firstTri = false;
			} else
			{
				aabb->join(bbox);
			}

			//tris.back()->parent = this->parent;
		}
	}

	//add to offset for use in the next object imported by GeometryObj::loadOBJ
	vertexNum = vertexOffset + vertices.size();
	uvNum = uvOffset + uvCoords.size();

	//set up object bounding box
	if(ifs.bad())
		return false;

	ifs.close();

	return true;
}
