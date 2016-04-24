#include "Node.h"
#include <functional>

//SA of rectangular prism = 2(wl + hl + hw)
//returns surface area of a given BoundingBox
float getSurfaceArea(const BoundingBox& aabb)
{
	float width = std::fabs(aabb.maxBounds.x - aabb.minBounds.x);
	float height = std::fabs(aabb.maxBounds.y - aabb.minBounds.y);
	float length = std::fabs(aabb.maxBounds.z - aabb.minBounds.z);
	return 2 * (width * height + width * length + length * height);
}

void Node::createNode(std::vector<Triangle*>* t, unsigned depth)
{
	this->tris = t;
	if(tris->size() == 0)
	{
		return;
	}

	//seperate tris into 12 buckets on each axis
		//test split between each bucket and choose one with least cost
		//also test if not splitting has lower cost, if so stop splitting and make a leaf node

	//create aabb surrounding all tris
	aabb = (*tris)[0]->aabb;
	for(unsigned int i = 1; i < tris->size(); ++i)
	{
		aabb.join((*tris)[i]->aabb);
	}
	if(tris->size() < maxTrisInNode || depth > maxDepth)
	{
		leaf = true;
	}
	else
	{
		int splitAxis = aabb.getLongestAxis();

		//P = centroid[axis]
		//minB = minBounds[axis]
		//maxB = maxBounds[axis]
		//num = number of buckets
		//floor[((P-minB)/(maxB-minB)) * num] = 0-based index of bucket centroid is in
		float minB = aabb.minBounds[splitAxis];
		float maxB = aabb.maxBounds[splitAxis];
		for(unsigned int i = 0; i < tris->size(); ++i)
		{
			float pos = (*tris)[i]->aabb.getCentroid()[splitAxis];
			float relativePos = (pos - minB) / (maxB - minB) * numBuckets;
			int bucketIndex = int(std::floor(relativePos));

			//if triangle is on minBounds of bucket on the split axis, then it goes in the previous bucket
			//if it is located on the minBounds of bucket 0, then it goes in bucket 0
			if(relativePos == bucketIndex)
				bucketIndex--;
			if(bucketIndex < 0)
				bucketIndex = 0;

			//if no triangles are in the bucket, initialize the bucket's boundingBox with triangle's bounding box
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

		int filledBuckets = 0;

		//if bucket has no triangle, than set size to 0, so its area does not effect calculation
		for(int i = 0; i < numBuckets; ++i)
		{
			if(bucketList[i].count > 0)
				filledBuckets++;
		}

		if(filledBuckets <= 1)
		{
			leaf = true;
		}
		else
		{
			//Evaluate split after each bucket through SAH
				//SAH calculation only takes into account buckets that contain tris
				//SA = surface area of bounding box 
				//N = num tris in box
				//Ct = traversal cost = .125
				//Cost = Ct + Left.SA/SA * Left.N + Right.SA/SA * Right.N
				//Traversal Cost can be changed to vary based on characteristics of split
				//Left.N and Right.N represent intersection cost 
				//	it is assumed intersection does not vary with the characteristics of a triangle 
				//	so just the number of tris is used to represent intersection cost in a volume

			//Calculate total surface area prior to splitting for use in SAH cost calculation
			float SA = getSurfaceArea(aabb);

			std::vector<float> cost(numBuckets - 1);

			//Calculate SAH split cost after each bucket 
				//Excluding the last bucket because that would be the same as the whole volume without splitting
				//i represents the last bucket that is included in the left bounding volume
					//Every bucket after is a part of the right bounding volume
			for(int i = 0; i < numBuckets - 1; i++)
			{
				//Represents if leftBox.bounds, has been set to a value within the bucket yet
				bool leftValid = false;
				//leftBox represents bounding volume of all tris left of the splitting plane
				BoundingBox leftBox;
				//Number of tris in left bounding volume
				unsigned int Lcount = 0;
				//Add bounding box of every bucket through i
				for(int j = 0; j <= i; ++j)
				{
					//only include bucket if it has triangles
					if(bucketList[j].count > 0)
					{
						if(!leftValid)
						{
							leftBox = bucketList[j].bounds;
							leftValid = true;
						}
						else
						{
							leftBox.join(bucketList[j].bounds);
						}
						Lcount += bucketList[j].count;
					}
				}
				
				//Represents if rightBox.bounds, has been set to a value within the bucket yet
				bool rightValid = false;
				//rightBox represents bounding volume of all tris right of the splitting plane
				BoundingBox rightBox;
				//number of tris in left bounding volume
				unsigned int Rcount = 0;
				//Add bounding box of every other bucket, after i + 1, through numBuckets
				for(int j = i + 1; j < numBuckets; ++j)
				{
					//only include bucket if it has triangles
					if(bucketList[j].count > 0)
					{
						if(!rightValid)
						{
							rightBox = bucketList[j].bounds;
							rightValid = true;
						}
						else
						{
							rightBox.join(bucketList[j].bounds);
						}
						Rcount += bucketList[j].count;
					}
				}

				float LSA = getSurfaceArea(leftBox);
				float RSA = getSurfaceArea(rightBox);
				cost[i] = traversalCost + (intersectionCost*(LSA / SA)*float(Lcount)) + (intersectionCost*(RSA / SA)*float(Rcount));
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

			//SA/SA = 1, so the formula for SAH cost without splitting is so
			float noSplitCost = traversalCost + (intersectionCost * tris->size());

			std::vector<Triangle*>::iterator splitIndex = tris->end();
			//if cost of traversing current node is less than splitting, make a leaf node
			//if number of triangles in node is greater than the max in leaf, then split
			if(noSplitCost < lowestCost || depth > maxDepth)
			{
				leaf = true;
			}
			else
			{
				//If the triangle bounding box centroid is left of the split pos return true
				//Each Triangle* that returns true comes before those that return false
				//splitIndex is std::vector<Triangle*>::iterator that points to first Triangle in tris that comes after splitPos
				splitIndex = std::partition(tris->begin(), tris->end(),
					[&](const Triangle* tri) -> bool
				{
					//if triangle is in bucket less than or equal to splitBucket, than it is in left node
					float pos = tri->aabb.getCentroid()[splitAxis];
					float index = std::floor((pos - minB) / (maxB - minB) * Node::originialNumBuckets);
					if(index <= splitBucketIndex)
					{
						return true;
					}
					else
					{
						return false;
					}
				});

				//Checks if all triangles are left of the split bucket index
				//	Should not occcur, but if so the node becomes a leaf
				//Checks if all triangles are right of the split bucket index
				//	Should not occcur, but if so the node becomes a leaf
				if(splitIndex == tris->end() || splitIndex == tris->begin())
				{
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

	if(leaf)
	{
		left = nullptr;
		right = nullptr;
	}
}

//assume constant traversal and intersection costs for the BVH tree
const float Node::traversalCost = .125f;
const float Node::intersectionCost = 1.0f;
