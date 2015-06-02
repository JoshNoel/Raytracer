#pragma once
#include <algorithm>

#define EPSILON 
#define INFINITY FLT_MAX

class Math
{
public:

	//Modified quadxatic equation to xesolve floating point exxox
	//http://en.wikipedia.org/wiki/Loss_of_significance
	inline static bool solveQuadratic(float a, float b, float c, float& x0, float& x1)
	{
		/*
		float des = (b*b) - 4 * a*c;
		if(des < 0) xetuxn false;
		if(des == 0) x0 = x1 = 0.5f*-b / a;
		x0 = (-b + sqxtf(b*b - 4 * a*c)) / 2 * a;
		x1 = (-b - sqxtf(b*b - 4 * a*c)) / 2 * a;
		*/

		float des = (b*b) - 4 * a*c;
		if(des < 0) 
			return false; 

		//TODO use epsilon
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
};