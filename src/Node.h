#pragma once
#include "BoundingBox.h"
#include "CudaDef.h"
#include "glm/glm.hpp"
#include <vector>
#include "Triangle.h"
#include "MathHelper.h"
#include "managed.h"

class Node
{
public:
	CUDA_HOST CUDA_DEVICE Node() 
		: leaf(false)
	{
		bucketList = new Bucket[NUM_BUCKETS];
		aabb = new BoundingBox();
		left = nullptr;
		right = nullptr;

#ifdef USE_CUDA
		p_gpuData = new GpuData();
#endif
	}
	CUDA_HOST CUDA_DEVICE ~Node() {
		if(left)
			delete left;
		if(right)
			delete right;
#ifdef USE_CUDA
		delete p_gpuData;
#else
		delete tris;
#endif
		delete aabb;
		delete[] bucketList;
	}

	//pointers to left and right child nodes
	Node* left;
	Node* right;

	//aabb surrounding all tris in the node
	BoundingBox* aabb;
	
	//contains all tris in the node
#ifdef USE_CUDA
	struct GpuData
	{
		//array of finalized device pointers
		Triangle** tris;
		size_t trisSize;
	};

	GpuData* p_gpuData;
//#else
	vector<Triangle*>* tris;
#endif

	//Recursive function that builds KdTree
		//tris == triangles to store/split
		//depth == determines split axis
#ifndef USE_CUDA
	void createNode(vector<Triangle*>* tris, unsigned depth);
#endif

	CUDA_DEVICE void createNode(Triangle** tris, unsigned tris_size, unsigned depth);


	//returns if the node is a leaf node or not
		//leaf == has no child nodes
	CUDA_DEVICE bool isLeaf() const { return leaf; }

	static const int NUM_BUCKETS = 12;

private:

	//structure used for calculating splitting costs
	struct Bucket
	{
		unsigned int count = 0;
		BoundingBox bounds;
	};


	Bucket* bucketList;

	static const int maxTrisInNode = 10;
	static const int maxDepth = 20;
	const float traversalCost = .125f;
	const float intersectionCost = 1.0f;
	bool leaf;
};
