#ifndef BLENDER_WRAPPER_H
#define BLENDER_WRAPPER_H

#include <string>

class CBaseBlender;
class Timer;
class __declspec(dllexport) 
	CBlenderWrapper
{
public:
	enum DEVICE_TYPE{
		CPU,
		OPENCL,
		CUDA
	};
	explicit CBlenderWrapper() = default;
	~CBlenderWrapper();

public:
	void capabilityAssessment();
	CBaseBlender* getSingleInstance();
	void runImageBlender(unsigned int input_width, unsigned int input_height, unsigned int output_width, unsigned int output_height);
	void ReleaseResources();

private:
	bool resetParameters(unsigned int width, unsigned int height, std::string offset);
	bool isSupportCUDA();
	bool isSupportOpenCL();

private:
	CBaseBlender* m_blender;
	std::string m_offset;
	unsigned int m_width;
	unsigned int m_height;
	DEVICE_TYPE m_deviceType;
	Timer m_profileTimer;
};

#endif




