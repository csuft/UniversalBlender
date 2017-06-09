#ifndef CPU_BLENDER_H
#define CPU_BLENDER_H

#include "../base/BaseBlender.h"

class CCPUBlender : public CBaseBlender
{
public:
	CCPUBlender();
	CCPUBlender(int mode);
	~CCPUBlender();
	 
	virtual void runBlender(unsigned char* input_data, unsigned char* output_data);
	virtual void destroyBlender();
};

#endif



