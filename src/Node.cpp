#include "Node.h"
#include <functional>

void Node::createNode(std::vector<Triangle*>* t, unsigned depth)
{
	this->tris = t;
	if(tris->size() == 0)
	{
		return;
	}
	//create aabb surrounding all tris,
	//seperate tris into 12 buckets on each axis
	aabb = (*tris)[0]->aabb;
	for(int i = 1; i < tris->size(); ++i)
	{
		aabb.join((*tris)[i]->aabb);
	}
	if(tris->size() > maxTrisInNode && depth < maxDepth)
	{
		int splitAxis = aabb.getLongestAxis();
		std::vector<int> bucketNums;

		//P = centroid[axis]
		//minB = minBounds[axis]
		//maxB = maxBounds[axis]
		//num = number of buckets
		//floor[((P-minB)/(maxB-minB)) * num] = 0-based index of bucket centroid is in
		float minB = aabb.minBounds[splitAxis];
		float maxB = aabb.maxBounds[splitAxis];
		for(int i = 0; i < tris->size(); ++i)
		{
			float pos = (*tris)[i]->aabb.getCentroid()[splitAxis];
			float relativePos = (pos - minB) / (maxB - minB) * numBuckets;
			int bucketIndex = std::floorf(relativePos);

			//if triangle is on minBounds of bucket on the split axis, then it goes in the previous bucket
			//if it is located on the minBounds of bucket 0, then it goes in bucket 0
			if(relativePos == bucketIndex)
				bucketIndex--;
			if(bucketIndex < 0)
				bucketIndex = 0;

			if(std::find(bucketNums.begin(), bucketNums.end(), bucketIndex) == bucketNums.end())
			{
				bucketNums.push_back(bucketIndex);
			}

			if(bucketList[bucketIndex].count == 0)
			{
				bucketList[bucketIndex].bounds = (*tris)[i]->aabb;
			}
			else
			{
				bucketList[bucketIndex].bounds.join((*tris)[i]->aabb);
			}

			bucketList[bucketIndex].count++;
		}

		std::sort(bucketNums.begin(), bucketNums.end());

		//if bucket has no triangle, than set size to 0, so its area does not effect calculation
		for(int i = 0; i < numBuckets; ++i)
		{
			if(bucketList[i].count == 0)
			{
				bucketList.erase(bucketList.begin() + i);
				--i;
				bucketList.shrink_to_fit();
				numBuckets = bucketList.size();
			}
		}

		if(numBuckets > 1)
		{
			//Evaluate split after each bucket through SAH
			//SAH calculation only takes into account buckets that contain tris
			//SA = surface area of bounding box 
			//N = num tris in box
			//Ct = traversal cost = .125
			//Cost = Ct + Left.SA/SA * Left.N + Right.SA/SA * Right.N
			//Traversal Cost can be changed to vary based on characteristics of split
			//Left.N and Right.N represent intersection cost that does not vary with each triangle, 
			//so just the number of tris is used
			//SA of rectangular prism = 2(wl + hl + hw)
			float width = std::abs(aabb.maxBounds.x - aabb.minBounds.x);
			float height = std::abs(aabb.maxBounds.y - aabb.minBounds.y);
			float length = std::abs(aabb.minBounds.z - aabb.maxBounds.z);
			float SA = 2 * (width * height + width * length + length * height);
			std::vector<float> cost(numBuckets - 1);
			for(int i = 0; i < numBuckets - 1; i++)
			{
				float Lwidth = std::fabsf(bucketList[i].bounds.maxBounds.x - bucketList[0].bounds.minBounds.x);
				float Lheight = std::fabsf(bucketList[i].bounds.maxBounds.y - bucketList[0].bounds.minBounds.y);
				float Llength = std::fabsf(bucketList[0].bounds.minBounds.z - bucketList[i].bounds.maxBounds.z);

				unsigned int Lcount = 0;
				for(int j = 0; j <= i; j++)
				{
					Lcount += bucketList[j].count;
				}

				float Rwidth = std::fabsf(bucketList[numBuckets - 1].bounds.maxBounds.x - bucketList[i + 1].bounds.minBounds.x);
				float Rheight = std::fabsf(bucketList[numBuckets - 1].bounds.maxBounds.y - bucketList[i + 1].bounds.minBounds.y);
				float Rlength = std::fabsf(bucketList[i + 1].bounds.minBounds.z - bucketList[numBuckets - 1].bounds.maxBounds.z);
				unsigned int Rcount = 0;
				for(int j = i + 1; j < numBuckets; j++)
				{
					Rcount += bucketList[j].count;
				}

				float LSA = 2.0f * (Lwidth*Llength + Lheight*Llength + Lheight*Lwidth);
				float RSA = 2.0f * (Rwidth*Rlength + Rheight*Rlength + Rheight*Rwidth);
				cost[i] = traversalCost + intersectionCost*(LSA / SA)*Lcount + intersectionCost*(RSA / SA)*Rcount;
			}

			//choose split with least cost
			int splitBucketIndex = 0;
			float lowestCost = cost[0];
			for(int i = 1; i < numBuckets - 1; i++)
			{
				if(cost[i] < lowestCost)
				{
					lowestCost = cost[i];
					splitBucketIndex = i;
				}
			}


			float noSplitCost = intersectionCost * tris->size();

			std::vector<Triangle*>::iterator splitIndex = tris->end();
			//if cost of traversing current node is less than splitting, make a leaf node
			//if number of triangles in node is greater than the max in leaf, then split
			if(noSplitCost > lowestCost && depth < maxDepth)
			{
				//float splitPos = bucketList[splitBucketIndex].bounds.maxBounds[splitAxis];
				int j = 0;

				//If the triangle bounding box centroid is left of the split pos return true
				//Each Triangle* that returns true comes before those that return false
				//splitIndex is std::vector<Triangle*>::iterator that points to first Triangle in tris that comes after splitPos
				splitIndex = std::partition(tris->begin(), tris->end(),
					[&](const Triangle* tri) -> bool
				{
					//if triangle is in bucket less than or equal to splitBucket, than it is in left node
					float pos = tri->aabb.getCentroid()[splitAxis];
					float index = std::floorf((pos - minB) / (maxB - minB) * Node::originialNumBuckets);
					if(index <= bucketNums[splitBucketIndex])
					{
						return true;
					}
					else //| 0 | 1 | 2 | 3 | 4 |
					{
						return false;
					}
					/*if(splitAxis != 2)
					{
						if(tri->aabb.getCentroid()[splitAxis] < splitPos)
							return true;
						else
							return false;
					}
					else
					{
						if(tri->aabb.getCentroid()[splitAxis] > splitPos)
							return true;
						else
							return false;
					}*/
				});
				if(splitIndex == tris->end())
				{
					left = new Node();
					left->createNode(new std::vector<Triangle*>(tris->begin(), splitIndex), depth + 1);
					leaf = true;
				}
				else if(splitIndex == tris->begin())
				{
					right = new Node();
					right->createNode(new std::vector<Triangle*>(splitIndex, tris->end()), depth + 1);
					leaf = true;
				}
				else
				{
					left = new Node();
					right = new Node();
					left->createNode(new std::vector<Triangle*>(tris->begin(), splitIndex), depth + 1);
					right->createNode(new std::vector<Triangle*>(splitIndex, tris->end()), depth + 1);
				}
			}
		}
	}
	else
	{
		leaf = true;
		left = nullptr;
		right = nullptr;
	}
}

const float Node::traversalCost = .125f;
const float Node::intersectionCost = 1.0f;