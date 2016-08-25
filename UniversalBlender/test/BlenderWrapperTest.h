#ifndef _BLENDER_WRAPPER_H
#define _BLENDER_WRAPPER_H

#include "headers.h"
#include "../wrapper/BlenderWrapper.h"

class CBlenderWrapperTest : public CppUnit::TestFixture, public CBlenderWrapper
{
public:
	CBlenderWrapperTest();
	~CBlenderWrapperTest();
};




#endif




