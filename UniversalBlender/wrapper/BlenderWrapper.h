#ifndef BLENDER_WRAPPER_H
#define BLENDER_WRAPPER_H

#include <string>
#include <mutex>
#include <atomic>

typedef struct _BlenderParams {
	unsigned int	input_width		= 0;
	unsigned int	input_height	= 0;
	unsigned int	output_width	= 0;
	unsigned int	output_height	= 0;
	unsigned char*	input_data		= nullptr;
	unsigned char*	output_data		= nullptr;
	std::string		offset			= "";
}BlenderParams, *BlenderParamsPtr;

class CBaseBlender;
class Timer;
class __declspec(dllexport) 
	CBlenderWrapper
{
public: 
	// 优先使用CUDA计算，再尝试OpenCL计算,最后考虑CPU计算
	enum BLENDER_DEVICE_TYPE {
		CUDA_BLENDER,
		OPENCL_BLENDER,
		CPU_BLENDER
	};

	// 1:双鱼眼全景展开map， 2:180度3d展开map， 3：双鱼眼全景展开map，且都位于中间
	enum BLENDER_TYPE {
		PANORAMIC_BLENDER			= 1,
		THREEDIMENSION_BLENDER		= 2,
		PANORAMIC_CENTER_BLENDER	= 3
	};
	explicit CBlenderWrapper();
	~CBlenderWrapper();

public:
	void capabilityAssessment();
	CBaseBlender* getSingleInstance();
	void runImageBlender(BlenderParams& params, BLENDER_TYPE type);

private:
	bool checkParameters(BlenderParams& params);
	bool isSupportCUDA();
	bool isSupportOpenCL();

private: 
	std::string m_offset;
	BLENDER_DEVICE_TYPE m_deviceType;
	std::mutex m_mutex;
	std::atomic<CBaseBlender*> m_blender;
};

#endif




