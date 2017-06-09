#include "OpenCLBlender.h"


COpenCLBlender::COpenCLBlender() : COpenCLBlender(2)
{

}

COpenCLBlender::COpenCLBlender(int mode) : m_inputImageSize(0), m_outputImageSize(0), m_widthFactor(0), m_heightFactor(0),  m_openclContext(nullptr),
m_program(nullptr), m_inputBuffer(nullptr), m_inputImage(nullptr), m_outputBuffer(nullptr), m_outputImage(nullptr), 
m_inputParamsBuffer(nullptr), m_leftMapBuffer(nullptr), m_rightMapBuffer(nullptr), m_commandQueue(nullptr), m_kernel(nullptr)
{
	memset(m_origins, 0, sizeof(int)* 3);
	memset(m_inputParams, 0, sizeof(int)* 16);
	m_unrollMap = new UnrollMap;
	m_colorMode = mode;
	m_channels = 4;
}


COpenCLBlender::~COpenCLBlender()
{
	if (m_openclContext != nullptr)
	{
		delete m_openclContext;
		m_openclContext = nullptr;
	}

	if (m_commandQueue != nullptr)
	{
		delete m_commandQueue;
		m_commandQueue = nullptr;
	}

	if (m_program != nullptr)
	{
		delete m_program;
		m_program = nullptr;
	}

	if (m_kernel != nullptr)
	{
		delete m_kernel;
		m_kernel = nullptr;
	}

	destroyBlender();
} 

/** 
 * 因为OpenCL要求work group的Range上下限都要是16的倍数，当图片的宽高不是16的倍数时，需要找到一个最大不大于宽高的16的倍数。
 * 否则会出现OUT_OF_RESOURCE或者INVALID_WORK_GROUP_SIZE异常
 */
int COpenCLBlender::findNearestNumber(unsigned int max)
{
	const int FACTOR = 16;
	int result = 0;
	while (result <= max)
	{
		result += FACTOR;
	}
	return (result - FACTOR);
}

void COpenCLBlender::runBlender(unsigned char* input_data, unsigned char* output_data)
{
	cl_int err;
	cl::Event event;

	// 3通道进3通道出，由于CUDA使用Texture来对map进行计算，只支持4通道图像，因此需要
	// 先将3通道图像转换为4通道图像，计算完成之后再转换为3通道图像。其余情况类似。
	unsigned char* inBuffer = nullptr;
	unsigned char* outBuffer = nullptr;
	startTimer();
	if (m_colorMode == 1)
	{
		outBuffer = new unsigned char[m_outputWidth*m_outputHeight * m_channels];
		inBuffer = new unsigned char[m_inputWidth*m_inputHeight * m_channels];
		RGB2RGBA(inBuffer, input_data, m_inputWidth*m_inputHeight * 3);
	}
	else if (m_colorMode == 2)
	{
		inBuffer = input_data;
		outBuffer = output_data;
	}
	else if (m_colorMode == 3)
	{
		inBuffer = new unsigned char[m_inputWidth*m_inputHeight * m_channels];
		RGB2RGBA(inBuffer, input_data, m_inputWidth*m_inputHeight * 3);
		outBuffer = output_data;
	}
	else
	{
		inBuffer = input_data;
		outBuffer = new unsigned char[m_outputWidth*m_outputHeight * m_channels];
	}
	stopTimer("Color Model Transform");
	startTimer();
	// Step 8: Run the kernels
	// parameters need to be fixed.
	err = m_commandQueue->enqueueWriteImage(*m_inputImage, CL_TRUE, m_origins, m_inputRegions, 0, 0, inBuffer, nullptr, &event);
	err = m_commandQueue->enqueueNDRangeKernel(*m_kernel, cl::NullRange, cl::NDRange(m_outputWidth, m_nearestNum), cl::NDRange(16, 16), nullptr, &event);
	checkError(err, "CommandQueue::enqueueNDRangeKernel()");
	event.wait();
	err = m_commandQueue->enqueueReadImage(*m_outputImage, CL_TRUE, m_origins, m_outputRegions, 0, 0, outBuffer, 0, &event);
	checkError(err, "CommandQueue::enqueueReadImage()");
	stopTimer("OpenCL Frame Mapping");
	if (m_colorMode == 1)      // THREE_CHANNELS
	{
		RGBA2RGB(output_data, outBuffer, m_outputWidth*m_outputHeight*m_channels);
		delete[] outBuffer;
		delete[] inBuffer;
	}
	else if (m_colorMode == 2)
	{
		// Nothing to do
	}
	else if (m_colorMode == 3)
	{
		delete[] inBuffer;
	}
	else
	{
		RGBA2RGB(output_data, outBuffer, m_outputWidth*m_outputHeight*m_channels);
		delete[] outBuffer;
	}
}

void COpenCLBlender::destroyBlender()
{
	if (m_inputImage != nullptr)
	{
		delete m_inputImage;
		m_inputImage = nullptr;
	}

	if (m_outputImage != nullptr)
	{
		delete m_outputImage;
		m_outputImage = nullptr;
	}

	if (m_inputParamsBuffer != nullptr)
	{
		delete m_inputParamsBuffer;
		m_inputParamsBuffer = nullptr;
	}

	if (m_leftMapBuffer != nullptr)
	{
		delete m_leftMapBuffer;
		m_leftMapBuffer = nullptr;
	}

	if (m_rightMapBuffer != nullptr)
	{
		delete m_rightMapBuffer;
		m_rightMapBuffer = nullptr;
	}

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

void COpenCLBlender::setupBlender()
{
	cl_int err;
	if (m_paramsChanged)
	{
		destroyBlender(); 
		m_unrollMap = new UnrollMap;
		if (m_blenderType == 1)
		{
			m_unrollMap->setOffset(m_offset); 
			m_unrollMap->init(m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight);
			m_leftMapData = m_unrollMap->getMap(0);
			m_rightMapData = m_unrollMap->getMap(1);
			m_nearestNum = m_outputHeight;
		}
		else if (m_blenderType == 2)
		{
			m_unrollMap->setOffset(m_offset, 200);
			m_unrollMap->init(m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight, 3);
			m_leftMapData = m_unrollMap->getMap(0);
			m_rightMapData = m_unrollMap->getMap(1);
			m_nearestNum = m_outputHeight;
		}
		else if (m_blenderType == 4)
		{
			m_unrollMap->setOffset(m_offset);
			m_unrollMap->init(m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight, 4);
			m_leftMapData = m_unrollMap->get3DMap();
			m_nearestNum = m_outputHeight;
		}
		else
		{
			m_unrollMap->setOffset(m_offset);
			m_unrollMap->init(m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight);
			m_leftMapData = m_unrollMap->getCylinderMap(0);
			m_rightMapData = m_unrollMap->getCylinderMap(1);
			m_nearestNum = findNearestNumber(m_outputHeight); 
		}
		 
		m_paramsChanged = false;

		// Step 7: Create buffers for kernels
		cl_channel_order image_channel_order = m_channels > 3 ? CL_RGBA : CL_RGB;
		cl_channel_type image_channel_data_type = m_channels > 3 ? CL_UNORM_INT8 : CL_UNORM_SHORT_565;

		cl::ImageFormat image_format(image_channel_order, image_channel_data_type);
		m_inputBuffer = new unsigned char[m_inputImageSize * m_channels]; 
		m_inputImage = new cl::Image2D(*m_openclContext, CL_MEM_USE_HOST_PTR, image_format, m_inputWidth, m_inputHeight, 0, m_inputBuffer, &err);
		checkError(err, "cl::Image2D()");
		m_outputBuffer = new unsigned char[m_outputImageSize * m_channels]; 
		m_outputImage = new cl::Image2D(*m_openclContext, CL_MEM_USE_HOST_PTR, image_format, m_outputWidth, m_outputHeight, 0, m_outputBuffer, &err);
		checkError(err, "cl::Image2D()");
		m_inputParamsBuffer = new cl::Buffer(*m_openclContext, CL_MEM_COPY_HOST_PTR, sizeof(int)* 16, m_inputParams, &err);
		m_leftMapBuffer = new cl::Buffer(*m_openclContext, CL_MEM_COPY_HOST_PTR, sizeof(float)*m_outputImageSize * 2, m_leftMapData, &err);
		m_rightMapBuffer = new cl::Buffer(*m_openclContext, CL_MEM_USE_HOST_PTR, sizeof(float)*m_outputImageSize * 2, m_rightMapData, &err);

		err = m_kernel->setArg(0, *m_inputImage);
		err |= m_kernel->setArg(1, *m_outputImage);
		err |= m_kernel->setArg(2, *m_leftMapBuffer);
		err |= m_kernel->setArg(3, *m_rightMapBuffer);
		err |= m_kernel->setArg(4, *m_inputParamsBuffer);
		err |= m_kernel->setArg(5, m_widthFactor);
		err |= m_kernel->setArg(6, m_heightFactor);
		err |= m_kernel->setArg(7, m_blenderType); 
		checkError(err, "Kernel::setArg()");
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

		m_widthFactor = 1.0f / m_inputWidth;
		m_heightFactor = 1.0f / m_inputHeight;

		int blenderWidth = 5 * m_outputWidth / 360;
		m_inputParams[0] = (m_outputWidth >> 2) - (blenderWidth >> 1);      // Left Blender Start
		m_inputParams[1] = m_inputParams[0] + blenderWidth;					// Left Blender End
		m_inputParams[2] = (m_outputWidth * 3 / 4) - (blenderWidth >> 1);   // Right Blender Start
		m_inputParams[3] = m_inputParams[2] + blenderWidth;                 // Right Blender End
		m_inputParams[4] = m_outputWidth * 2;
		m_inputParams[5] = m_outputHeight * 2;

		// To indicate the parameters have changed.
		m_paramsChanged = true;
	}
	m_blenderType = type;

	return true;
}

bool COpenCLBlender::initializeDevice()
{
	cl_int err; 
	int index;
	bool ret;
	// Step 1: Get platform list
	std::vector<cl::Platform> platformList;
	cl::Platform::get(&platformList);
	ret = checkError(platformList.size() != 0 ? CL_SUCCESS : -1, "cl::Platform::get");
	if (!ret)
	{
		return false;
	}
	std::string platformVersion;
	for (index = 0; index < platformList.size(); ++index)
	{ 
		err = platformList[index].getInfo((cl_platform_info)CL_PLATFORM_VERSION, &platformVersion);

		// Step 2: Create context for specific device type.
		cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[index])(), 0 };
		m_openclContext = new cl::Context(CL_DEVICE_TYPE_GPU, cprops, NULL, NULL, &err);
		if (err == CL_SUCCESS)
		{
			break;
		}
		
	}  
	if (index == platformList.size())
	{
		LOGERR("No platform can be used to create OpenCL context!");
		return false;
	}

	// Step 3: Get a list of available devices
	std::vector<cl::Device> devices;
	devices = m_openclContext->getInfo<CL_CONTEXT_DEVICES>();
	ret = checkError(devices.size() > 0 ? CL_SUCCESS : -1, "devices.size() > 0"); 
	if (!ret)
	{
		return false;
	}
	// Step 4: Create source and program object
	cl::Program::Sources source(1, std::make_pair(BLEND_KERNEL_STRING.c_str(), BLEND_KERNEL_STRING.length() + 1));
	m_program = new cl::Program(*m_openclContext, source);
	err = m_program->build(devices, "");
	ret = checkError(err, "Program::Build()");
	if (!ret)
	{
		return false;
	}

	// Step 5: Create kernel object and set arguments
	m_kernel = new cl::Kernel(*m_program, "opencl_blend", &err);
	ret = checkError(err, "Kernel::Kernel()");
	if (!ret)
	{
		return false;
	}

	// Step 6: Create a one-to-one mapping command queue and execute command
	m_commandQueue = new cl::CommandQueue(*m_openclContext, devices[0], 0, &err);
	ret = checkError(err, "CommandQueue::CommandQueue()"); 
	
	return ret;
}

bool COpenCLBlender::checkError(cl_int err, const char* name)
{ 
	if (err != CL_SUCCESS)
	{
		LOGERR("Error: %s, Code: [%s]", name, get_error_string(err));
		return false;
	}

	return true;
}

const char * COpenCLBlender::get_error_string(cl_int err){
	switch (err){
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
	default: return "Unknown OpenCL error";
	}
}