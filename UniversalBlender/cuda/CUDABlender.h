#ifndef CUDA_BLENDER_H
#define CUDA_BLENDER_H

#include "../base/BaseBlender.h"

class CCUDABlender : public CBaseBlender
{
public:
	CCUDABlender();
	~CCUDABlender();

	virtual void setupBlender();
	virtual void runBlender(const unsigned char* input_data, unsigned char* output_data, int type);
	virtual void destroyBlender();
	virtual void setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned oh, std::string offset);

	// private data member
private:

	// private method member
private:

};


#endif

