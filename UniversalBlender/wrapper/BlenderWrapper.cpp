#include "BlenderWrapper.h"

#include "../base/BaseBlender.h"
#include "../utils/log.h"
#include "../utils/timer.h"
#include "../cuda/CUDABlender.h"
#include "../opencl/OpenCLBlender.h"
#include "../cpu/CPUBlender.h"

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

CBlenderWrapper::CBlenderWrapper() : m_blender(nullptr), m_deviceType(CPU_BLENDER), m_offset("")
{
}


CBlenderWrapper::~CBlenderWrapper()
{
	CBaseBlender* temp = m_blender.load(std::memory_order_relaxed);
	if (temp != nullptr)
	{
		delete temp;
		temp = nullptr;
	}
}

int CBlenderWrapper::capabilityAssessment()
{ 
	if (isSupportCUDA())
	{
		m_deviceType = CUDA_BLENDER;
		LOGINFO("CUDA compute technology is available in this platform.");
	}
	else if (isSupportOpenCL())
	{
		m_deviceType = OPENCL_BLENDER;
		LOGINFO("OpenCL compute technology is available in this platform.");
	}
	else
	{
		m_deviceType = CPU_BLENDER;
		LOGINFO("Only CPU is available in this platform.");
	}
	
	return m_deviceType;
}

// Determine whether the platform support CUDA.
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
			LOGINFO("CUDA Device Info:\nName: \t\t%s\nTotalGlobalMem:\t\t%uMB\nSharedMemPerBlock:\t%dKB\nRegsPerBlock:\t\t%d\nwarpSize:\t\t\t%d\nmemPitch:\t\t\t%d\nMaxThreadPerBlock:\t%d\nMaxThreadsDim:\t\tx = %d, y = %d,z =%d\nMaxGridSize: \t\tx = %d,y = %d,z = %d\nTotalConstMem:\t\t%d\nmajor:\t\t\t\t%d\nminor:\t\t\t\t5d\nTextureAlignment:\t%d\t", 
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

	return true;
}

// Determine whether the platform support OpenCL
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
		LOGINFO("Platform vendor: \t\t%s, version: \t\t%s, name: %s, profile: %s", platformVendor.c_str(), platformVersion.c_str(), platformName.c_str(), platformProfile.c_str());
	}

	return true;
}

// Implement singleton pattern using C++11 low level ordering constraints.
void CBlenderWrapper::getSingleInstance(COLOR_MODE mode)
{
	CBaseBlender* temp = m_blender.load(std::memory_order_acquire);
	if (temp == nullptr)
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		temp = m_blender.load(std::memory_order_relaxed);
		if (temp == nullptr)
		{
			if (CUDA_BLENDER == m_deviceType)
			{
				temp = new CCUDABlender(mode);
				LOGINFO("CUDA instance address: %p", temp);
			}
			else if (OPENCL_BLENDER == m_deviceType)
			{
				temp = new COpenCLBlender(mode);
				LOGINFO("OpenCL instance address: %p", temp);
			}
			else
			{
				temp = new CCPUBlender(mode);
				LOGINFO("CPU instance address: %p", temp);
			}
			m_blender.store(temp, std::memory_order_release);
		}
	} 
}

void CBlenderWrapper::initializeDevice()
{
	CCUDABlender* cudaBlender = nullptr;
	COpenCLBlender* openclBlender = nullptr;
	switch (m_deviceType)
	{
	case CBlenderWrapper::CUDA_BLENDER:
		cudaBlender = dynamic_cast<CCUDABlender*>(m_blender.load(std::memory_order_relaxed));
		if (cudaBlender)
		{
			cudaBlender->initializeDevice();
		}
		
		break;
	case CBlenderWrapper::OPENCL_BLENDER:
		openclBlender = dynamic_cast<COpenCLBlender*>(m_blender.load(std::memory_order_relaxed));
		if (openclBlender)
		{
			openclBlender->initializeDevice();
		}
		
		break;
	case CBlenderWrapper::CPU_BLENDER:
		// leave it alone
		break;
	default:
		break;
	}
}

// 该接口封装CPU/OPENCL/CUDA进行图像拼接渲染
void CBlenderWrapper::runImageBlender(BlenderParams& params, BLENDER_TYPE type)
{
	if (checkParameters(params))
	{
		CBaseBlender* blender = m_blender.load(std::memory_order_relaxed);
		bool retVal = blender->setParams(params.input_width, params.input_height, params.output_width, params.output_height, params.offset, type);
		if (retVal)
		{
			blender->setupBlender();
			blender->runBlender(params.input_data, params.output_data);
		}
	}
}

bool CBlenderWrapper::checkParameters(BlenderParams& params)
{
	if (params.input_width <= 0 || params.input_height <= 0 || params.input_data == nullptr)
	{
		LOGERR("Invalid input parameters, please check again carefully!");
		return false;
	}

	if (params.output_width <= 0 || params.output_height <= 0 || params.output_data == nullptr)
	{
		LOGERR("Invalid output parameters, please check again carefully!");
		return false;
	}

	if (params.offset.empty())
	{
		LOGERR("Offset is empty, please check again carefully!");
		return false;
	}

	return true;
}

