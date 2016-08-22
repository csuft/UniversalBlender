#ifndef CPU_BLENDER_H
#define CPU_BLENDER_H

#include "../base/BaseBlender.h"

class CCPUBlender : public CBaseBlender
{
public:
	CCPUBlender();
	CCPUBlender(int channels);
	~CCPUBlender();

	virtual void setupBlender();
	virtual void runBlender(unsigned char* input_data, unsigned char* output_data);
	virtual void destroyBlender();
	virtual bool setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned oh, std::string offset, int type);

private:
	unsigned char* addAlphaChannel(const unsigned char* inputImage);
	unsigned char* removeAlphaChannel(const unsigned char* inputImage);

};

#endif



