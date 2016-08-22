#ifndef CUDA_BLENDER_H
#define CUDA_BLENDER_H

#include "../base/BaseBlender.h"

class CCUDABlender : public CBaseBlender
{
public:
	CCUDABlender();
	~CCUDABlender();

	virtual void runBlender(unsigned char* input_data, unsigned char* output_data);
	virtual void destroyBlender();
	// private data member
private:

	// private method member
private:

};


#endif

