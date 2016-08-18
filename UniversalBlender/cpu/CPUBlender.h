#ifndef CPU_BLENDER_H
#define CPU_BLENDER_H

#include "../base/BaseBlender.h"

class CCPUBlender : public CBaseBlender
{
public:
	CCPUBlender();
	~CCPUBlender();

private:
	unsigned char* addAlphaChannel(const unsigned char* inputImage);
	unsigned char* removeAlphaChannel(const unsigned char* inputImage);
};

#endif



