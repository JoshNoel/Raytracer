#pragma once
#include <algorithm>

//used to create slight offset between secondary rays and intersection point
	//avoids self intersection
#ifndef RAY_EPSILON
	#define RAY_EPSILON 1e-3f;
#endif

//Used to define maximum bounds of scene
	//10,000 units in both directions on all axis
	//Each axis is 20,000 units in length
#ifndef _INFINITY
	#define _INFINITY 10000.0f
#endif

#define _PI_ 2.0f*std::acos(0.0f)

//converts degrees to radians
#define degToRad(d) d*(_PI_ / 180.0f)

class Math
{
public:

	//Modified quadratic equation to resolve floating point error
	//http://en.wikipedia.org/wiki/Loss_of_significance
	inline static bool solveQuadratic(float a, float b, float c, float& x0, float& x1)
	{
		float des = (b*b) - 4 * a*c;
		if(des < 0) 
			return false; 

		//if b==0 use the regular quadratic equation
		if(b == 0)
		{
			x0 = sqrtf(-4 * a*c) / (2 * a);
			if(des > 0)
				x1 = -sqrtf(-4 * a*c) / (2 * a);
			return true;
		}

		float val;
		if(b > 0)
			val = -b + sqrtf(des);
		else
			val = -b - sqrtf(des);

		x0 = 0.5f*val / a;
		if(des > 0)
			x1 = c / (a*x0);
		return true;
		
	}

	//returns sign of num
	//	negative -> return -1
	//	positive -> return 1
	template <typename T> 
	inline static int sign(T num)
	{
		return (num > 0) - (num < 0);
	}
};
