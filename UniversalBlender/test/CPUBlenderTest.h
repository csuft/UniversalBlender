#ifndef _CPU_BLENDER_H
#define _CPU_BLENDER_H

#include "headers.h"
#include "../cpu/CPUBlender.h"

class CCPUBlenderTest : public CppUnit::TestFixture , public CCPUBlender
{
public:
	CCPUBlenderTest();
	~CCPUBlenderTest();

	static CppUnit::Test* suite();

	void setUp(){}
	void tearDown(){}

protected:
	void test_isBase64Decoded();
	void test_isOffsetValid();

};

#endif

