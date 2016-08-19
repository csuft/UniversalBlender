#ifndef OPENCL_BLENDER_H
#define OPENCL_BLENDER_H

#include "../base/BaseBlender.h"

class COpenCLBlender : public CBaseBlender
{
public:
	COpenCLBlender();
	~COpenCLBlender();

	virtual void setupBlender();
	virtual void runBlender(const unsigned char* input_data, unsigned char* output_data, int type);
	virtual void destroyBlender();
	virtual void setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned oh, std::string offset);

	//private data member
private:

	// private method member
private:

};

#endif


