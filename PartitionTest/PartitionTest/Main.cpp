#include <vector>
#include <algorithm>

bool checkNum(int i)
{
	return i < 1;
}

void main()
{
	std::vector<int> numbers = { 1,2,3,4,5,6,7,8,9 };
	std::vector<int>::iterator it = numbers.end();
	it = std::partition(numbers.begin(), numbers.end(), checkNum);
	std::vector<int> nums1(numbers.begin(), it);
	std::vector<int> nums2(it, numbers.end());
	std::vector<int>::iterator ij = numbers.end();
	if(it == numbers.begin())
		int x = 4;
	int j = 0;
}