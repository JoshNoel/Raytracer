#pragma once
#include "Object.h"
#include "Ray.h"

class AccelerationStruct
{
protected:
	AccelerationStruct();
	virtual ~AccelerationStruct();

	virtual void initialize() = 0;
private:

};