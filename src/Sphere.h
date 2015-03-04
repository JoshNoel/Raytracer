#pragma once
#include "glm\glm.hpp"
#include "Object.h"
class Sphere
	: public Object
{
	/*
	*Sphere implicit=
	*pos = point position, C = sphere center, r = radius
	*Eq.1:	(pos-C)²=r²
	*Eq.2:	(x-C)²+(y-C)²+(z-C)²=r² 
	*Eq.3:	(pos-C)²-r²=0	|o-C + dt|->ray implicit for sphere
	*o = ray origin, t = variable, d = ray direction
	*Eq.4:	|o-C|²+(2|o-c|dt)+(dt)²-r²=0	|o-C|²+2|o-C|t+t²-r²=0
	*These relationships are true for any point on the sphere
	*If Eq.3>0 point is outside sphere
	*If Eq.3<0 point is inside sphere
	*If Eq.3=0 point is on sphere
	*Subsitute ray formula for pos creating eq.4
	*d is normalized, therefore d²=1(see note)
	*	t²+2|o-C|dt+|o-C|²-r²=0
	*	a=1
	*	b=2d|o-C|
	*	c=|o-C|²-r²
	*Find determinent to see if intersection exists
	*If 1 exists ray is tangent
	*If 2 exist the smaller t value(t₀) is the first intersection 
	*---------(-----)--------->
	*		  t₀	t₁
	*/
	///////////NOTE: vector²=vector·vector/////////////
public:
	Sphere();
	Sphere(glm::vec3 pos, float radius, Material mat = Material());
	~Sphere();
	OBJECT_TYPE getType() const override { return OBJECT_TYPE::SPHERE; }
	float radius;
};

