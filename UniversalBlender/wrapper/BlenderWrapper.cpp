#include "BlenderWrapper.h"

#include <vector>

#include "../base/BaseBlender.h"
#include "../utils/log.h"
#include "../utils/timer.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#include <OpenCL/cl_gl.h>
#else 
#include <CL/cl.hpp>
#include <CL/cl_gl.h>
#endif

#define check_CUDA_status(ret) \
if (ret != cudaSuccess) {\
	LOGERR("current thread id = %d, CUDA ERR:%s, error code = %d", m_profileTimer.PthreadSelf(), cudaGetErrorString(ret), ret); return ret; \
}

CBlenderWrapper::CBlenderWrapper() : m_blender(nullptr), m_deviceType(CPU)
{
}


CBlenderWrapper::~CBlenderWrapper()
{
}

void CBlenderWrapper::capabilityAssessment()
{

}

bool CBlenderWrapper::isSupportCUDA()
{
	cudaError_t retVal = cudaSuccess;
	int nDeviceCount, index;

	retVal = cudaGetDeviceCount(&nDeviceCount);
	if (cudaSuccess != retVal)
	{
		LOGERR("Error Desc:%s, Error code = %d", cudaGetErrorString(retVal), retVal);
		return false;
	}

	cudaDeviceProp deviceProps;
	for (index = 0; index < nDeviceCount; ++index)
	{
		retVal = cudaGetDeviceProperties(&deviceProps, index);
		if (cudaSuccess != retVal)
		{
			LOGERR("Error Desc:%s, Error code = %d", cudaGetErrorString(retVal), retVal);
			return false;
		}
		// Compute capability at least be 1.x
		if (deviceProps.major >= 1)
		{
			LOGINFO("CUDA Device Info:\nName: %s\nTotalGlobalMem:%u MB\nSharedMemPerBlock:%d KB\nRegsPerBlock : %d\nwarpSize:%d\nmemPitch: %d\nMaxThreadPerBlock:%d\nMaxThreadsDim:x = %d, y = %d,z =%d\nMaxGridSize: x = %d,y = %d,z = %d\nTotalConstMem : %d\nmajor:%d\nminor:5d\nTextureAlignment:%d\n", 
				deviceProps.name,
				deviceProps.totalGlobalMem / (1024 * 1024),
				deviceProps.sharedMemPerBlock / 1024,
				deviceProps.regsPerBlock,
				deviceProps.warpSize,
				deviceProps.memPitch,
				deviceProps.maxThreadsPerBlock,
				deviceProps.maxThreadsDim[0],
				deviceProps.maxThreadsDim[1],
				deviceProps.maxThreadsDim[2],
				deviceProps.maxGridSize[0],
				deviceProps.maxGridSize[1],
				deviceProps.maxGridSize[2],
				(int)deviceProps.totalConstMem,
				deviceProps.major,
				deviceProps.minor,
				(int)deviceProps.textureAlignment);
			break;
		}
	}

	if (index == nDeviceCount)
	{
		LOGERR("Could not find any available CUDA device!");
		return false;
	}

	retVal = cudaSetDevice(index);
	if (cudaSuccess != retVal)
	{
		LOGERR("Error Desc:%s, Error code = %d", cudaGetErrorString(retVal), retVal);
		return false;
	}

	return true;
}

bool CBlenderWrapper::isSupportOpenCL()
{
	cl_int err;
	std::vector<cl::Platform> platformList;
	err = cl::Platform::get(&platformList);
	if (CL_INVALID_VALUE == err || platformList.size() == 0)
	{
		LOGERR("Could not find any available OpenCL device.");
		return false;
	}

	std::string platformVendor;
	std::string platformVersion;
	std::string platformName;
	std::string platformProfile;
	for (int i = 0; i < platformList.size(); ++i)
	{
		err = platformList[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
		err = platformList[i].getInfo((cl_platform_info)CL_PLATFORM_PROFILE, &platformProfile);
		err = platformList[i].getInfo((cl_platform_info)CL_PLATFORM_NAME, &platformName);
		err = platformList[i].getInfo((cl_platform_info)CL_PLATFORM_VERSION, &platformVersion);
		LOGINFO("Platform vendor: %s, version: %s, name: %s, profile: %s", platformVendor.c_str(), platformVersion.c_str(), platformName.c_str(), platformProfile.c_str());
	}

	return true;
}

CBaseBlender* CBlenderWrapper::getSingleInstance()
{

}

void CBlenderWrapper::runImageBlender(unsigned int input_width, unsigned int input_height, unsigned int output_width, unsigned int output_height)
{

}

void CBlenderWrapper::ReleaseResources()
{

}

bool CBlenderWrapper::resetParameters(unsigned int width, unsigned int height, std::string offset)
{

}

