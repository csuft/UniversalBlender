#ifndef OPENCL_BLENDER_H
#define OPENCL_BLENDER_H

#include "../base/BaseBlender.h"

class COpenCLBlender : public CBaseBlender
{
public:
	COpenCLBlender();
	~COpenCLBlender();

	virtual void setupBlender();
	virtual void runBlender(unsigned char* input_data, unsigned char* output_data);
	virtual void destroyBlender();
	virtual bool setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned int oh, std::string offset, int type);

	//private data member
private:

	// private method member
private:

};

#endif


