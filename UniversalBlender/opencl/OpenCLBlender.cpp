#include "OpenCLBlender.h"


COpenCLBlender::COpenCLBlender() : COpenCLBlender(4)
{

}

COpenCLBlender::COpenCLBlender(int channels) : m_inputImageSize(0), m_outputImageSize(0), m_heightFactor(0), m_widthFactor(0)
{
	memset(m_origins, 0, sizeof(int)* 3);
	memset(m_inputParams, 0, sizeof(int)* 16);
	m_unrollMap = new UnrollMap;
	if (channels != 3 || channels != 4)
	{
		channels = 4;
	}
	m_channels = channels;
}


COpenCLBlender::~COpenCLBlender()
{
	destroyBlender();
} 

void COpenCLBlender::runBlender(unsigned char* input_data, unsigned char* output_data)
{
	cl_int err;
	cl::Event event;
	if (m_blenderType == 1)
	{
		// parameters need to be fixed.
		err = m_commandQueue.enqueueWriteImage(m_inputImage, CL_TRUE, m_origins, m_inputRegions, 0, 0, input_data, nullptr, &event);
		err = m_commandQueue.enqueueNDRangeKernel(m_kernel, cl::NullRange, cl::NDRange(2), cl::NDRange(16, 16), nullptr, &event);
		checkError(err, "CommandQueue::enqueueNDRangeKernel()");
		event.wait();
		err = m_commandQueue.enqueueReadImage(m_outputImage, CL_TRUE, m_origins, m_outputRegions, 0, 0, output_data, 0, &event);
		checkError(err, "CommandQueue::enqueueReadImage()");
	}
	else if (m_blenderType == 2)
	{

	}
	else
	{

	} 
	
}

void COpenCLBlender::destroyBlender()
{
	if (m_inputBuffer != nullptr)
	{
		delete []m_inputBuffer;
		m_inputBuffer = nullptr;
	}

	if (m_outputBuffer != nullptr)
	{
		delete[]m_outputBuffer;
		m_outputBuffer = nullptr;
	}

	if (nullptr != m_unrollMap)
	{
		delete m_unrollMap;
		m_unrollMap = nullptr;
	}
} 

bool COpenCLBlender::setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned oh, std::string offset, int type)
{
	if (iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0)
	{
		LOGERR("Invalid resolution parameters, please check again carefully!");
		return false;
	}

	if (!isOffsetValid(offset))
	{
		LOGERR("Invalid offset format, please check again carefully!");
		return false;
	}

	if (iw != m_inputWidth || ih != m_inputHeight || ow != m_outputWidth || oh != m_outputHeight || m_offset.compare(offset))
	{
		m_inputWidth = iw;
		m_inputHeight = ih;
		m_outputWidth = ow;
		m_outputHeight = oh;
		m_offset = offset;
		m_inputImageSize = m_inputWidth * m_inputHeight;
		m_outputImageSize = m_outputWidth * m_outputHeight;
		
		m_inputRegions[0] = m_inputWidth;
		m_inputRegions[1] = m_inputHeight;
		m_inputRegions[2] = 1;
		
		m_outputRegions[0] = m_outputWidth;
		m_outputRegions[1] = m_outputHeight;
		m_outputRegions[2] = 1;

		m_widthFactor = 1.0 / m_inputWidth;
		m_heightFactor = 1.0 / m_inputHeight;

		// To indicate the parameters have changed.
		m_paramsChanged = true;
	}

	return true;
}

void COpenCLBlender::initializeDevice()
{
	cl_int err;
	// Step 1: Get platform list
	std::vector<cl::Platform> platformList;
	cl::Platform::get(&platformList);
	checkError(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
	LOGINFO("Platform number is: ", platformList.size());
	// Output the platform information.
	std::string platformVendor;
	for (int i = 0; i < platformList.size(); ++i)
	{
		platformList[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &platformVendor);
		LOGINFO("Platform is by: ", platformVendor.c_str());
		platformList[i].getInfo((cl_platform_info)CL_PLATFORM_EXTENSIONS, &platformVendor);
		LOGINFO("Platform extension: ", platformVendor.c_str());
		platformList[i].getInfo((cl_platform_info)CL_PLATFORM_NAME, &platformVendor);
		LOGINFO("Platform name: ", platformVendor.c_str());
		platformList[i].getInfo((cl_platform_info)CL_PLATFORM_PROFILE, &platformVendor);
		LOGINFO("Platform profile: ", platformVendor.c_str());
		platformList[i].getInfo((cl_platform_info)CL_PLATFORM_VERSION, &platformVendor);
		LOGINFO("Platform version: ", platformVendor.c_str());
	}

	// Step 2: Create context for specific device type.
	cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0 };
	cl::Context context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, &err);
	checkError(err, "Context::Context()");
	m_openclContext = context;

	// Step 3: Create buffers for kernels
	cl_channel_order image_channel_order = m_channels > 3 ? CL_RGBA : CL_RGB;
	cl_channel_type image_channel_data_type = m_channels > 3 ? CL_UNORM_INT8 : CL_UNORM_SHORT_565;

	if (m_blenderType == 1)
	{ 
		cl::ImageFormat image_format(image_channel_order, image_channel_data_type);
		m_inputBuffer = new unsigned char[m_inputImageSize * m_channels];
		cl::Image2D inputImage(m_openclContext, CL_MEM_USE_HOST_PTR, image_format, m_inputWidth, m_inputHeight, 0, m_inputBuffer, &err);
		m_inputImage = inputImage;

		m_outputBuffer = new unsigned char[m_outputImageSize * m_channels];
		cl::Image2D outputImage(m_openclContext, CL_MEM_USE_HOST_PTR, image_format, m_outputWidth, m_outputHeight, 0, m_outputBuffer, &err);
		m_outputImage = outputImage;

		cl::Buffer paramsBuffer(m_openclContext, CL_MEM_COPY_HOST_PTR, sizeof(int)* 16, m_inputParams, &err);
		m_inputParamsBuffer = paramsBuffer;

		cl::Buffer leftMapBuffer(m_openclContext, CL_MEM_COPY_HOST_PTR, sizeof(float)*m_outputImageSize * 2, m_leftMapData, &err);
		m_leftMapBuffer = leftMapBuffer;

		cl::Buffer rightMapBuffer(m_openclContext, CL_MEM_USE_HOST_PTR, sizeof(float)*m_outputImageSize * 2, m_rightMapData, &err);
		m_rightMapBuffer = rightMapBuffer;
	}
	else if (m_blenderType == 2)
	{

	}
	else
	{

	}

	// Step 4: Get a list of available devices
	std::vector<cl::Device> devices;
	devices = m_openclContext.getInfo<CL_CONTEXT_DEVICES>();
	checkError(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0");

	// Step 5: Read kernel implementation file
	cl::Program::Sources source(1, std::make_pair(BLEND_KERNEL_STRING.c_str(), BLEND_KERNEL_STRING.length() + 1));
	cl::Program program(m_openclContext, source);
	err = program.build(devices, "");
	checkError(err, "Program::Build()");

	// Step 6: Create kernel object and set arguments
	cl::Kernel kernel(program, "opencl_blend", &err);
	checkError(err, "Kernel::Kernel()");
	m_kernel = kernel;
	err |= m_kernel.setArg(0, m_inputImage);
	err |= m_kernel.setArg(1, m_outputImage);
	err |= m_kernel.setArg(2, m_leftMapBuffer);
	err |= m_kernel.setArg(3, m_rightMapBuffer);
	err |= m_kernel.setArg(4, m_inputParamsBuffer);
	err |= m_kernel.setArg(5, m_widthFactor);
	err |= m_kernel.setArg(6, m_heightFactor);
	checkError(err, "Kernel::setArg()");

	// Step 7: Create a one-to-one mapping command queue and execute command
	cl::CommandQueue queue(m_openclContext, devices[0], 0, &err);
	checkError(err, "CommandQueue::CommandQueue()");
	m_commandQueue = queue;
}

inline bool COpenCLBlender::checkError(cl_int err, const char* name)
{
	if (err != CL_SUCCESS)
	{
		LOGERR("Error: %s, Code: %d", name, err);
		return false;
	}

	return true;
}