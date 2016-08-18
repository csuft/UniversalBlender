#ifndef OPENCL_BLENDER_H
#define OPENCL_BLENDER_H

#include "../base/BaseBlender.h"

class COpenCLBlender : public CBaseBlender
{
public:
	COpenCLBlender();
	~COpenCLBlender();

	virtual void setupBlender();
	virtual void runBlender();
	virtual void destroyBlender();

	//private data member
private:

	// private method member
private:

};

#endif


