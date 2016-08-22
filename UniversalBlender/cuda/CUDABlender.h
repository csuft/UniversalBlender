#ifndef CUDA_BLENDER_H
#define CUDA_BLENDER_H

#include "../base/BaseBlender.h"

class CCUDABlender : public CBaseBlender
{
public:
	CCUDABlender();
	~CCUDABlender();

	virtual void setupBlender();
	virtual void runBlender(unsigned char* input_data, unsigned char* output_data);
	virtual void destroyBlender();
	virtual bool setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned int oh, std::string offset, int type);

	// private data member
private:

	// private method member
private:

};


#endif

