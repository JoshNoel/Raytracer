#pragma once
#include "glm\glm.hpp"
#include "BoundingBox.h"
#include <vector>
#include "Triangle.h"
#include "MathHelper.h"

class Node
{
public:
	Node() 
		: aabb(), bucketList(numBuckets)
	{
		left = nullptr;
		right = nullptr;
	}
	~Node() {}

	//pointers to left and right child nodes
	Node* left;
	Node* right;

	//aabb surrounding all tris in the node
	BoundingBox aabb;
	
	//contains all tris in the node
	std::vector<Triangle*>* tris;

	//Recursive function that builds KdTree
		//tris == triangles to store/split
		//depth == determines split axis
	void createNode(std::vector<Triangle*>* tris, unsigned depth);

	//returns if the node is a leaf node or not
		//leaf == has no child nodes
	bool isLeaf() const { return leaf; }

	static const int originialNumBuckets = 12;

private:

	//structure used for calculating splitting costs
	struct Bucket
	{
		unsigned int count = 0;
		BoundingBox bounds;
	};

	int numBuckets = originialNumBuckets;

	std::vector<Bucket> bucketList;

	static const int maxTrisInNode = 10;
	static const int maxDepth = 20;
	static const float traversalCost;
	static const float intersectionCost;
	bool leaf;
};