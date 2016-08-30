#include "CUDABlender.h"

extern "C" cudaError_t cuFinishToBlender(cudaArray *inputBuffer, float *left_map, float*right_map, float* alpha_table, int image_width, int image_height, int bd_width, dim3 thread, dim3 numBlock, unsigned char *uOutBuffer, int type);

CCUDABlender::CCUDABlender() : CCUDABlender(4)
{
	
}

CCUDABlender::CCUDABlender(int channels) : m_inputImageSize(0), m_outputImageSize(0), m_cudaOutputBuffer(nullptr),
m_cudaAlphaTable(nullptr), m_cudaArray(nullptr), m_cudaLeftMapData(nullptr), m_cudaRightMapData(nullptr)
{
	m_unrollMap = new UnrollMap;
	if (channels != 3 || channels != 4)
	{
		channels = 4;
	}
	m_channels = channels;
	m_threadsPerBlock.x = 16;
	m_threadsPerBlock.y = 16;
	m_threadsPerBlock.z = 1;
}


CCUDABlender::~CCUDABlender()
{
	destroyBlender();
}

void CCUDABlender::setupBlender()
{
	cudaError_t err = cudaSuccess;

	if (m_paramsChanged)
	{
		destroyBlender();
		m_unrollMap = new UnrollMap;
		m_unrollMap->setOffset(m_offset);
		m_unrollMap->init(m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight, m_blenderType);
		m_leftMapData = m_unrollMap->getMap(0);
		m_rightMapData = m_unrollMap->getMap(1);
		m_paramsChanged = false;

		if (m_blenderType == 1)
		{
			err = cudaMalloc(&m_cudaLeftMapData, sizeof(float)*m_outputImageSize * 2);
			err = cudaMalloc(&m_cudaRightMapData, sizeof(float)*m_outputImageSize * 2);
			err = cudaMemcpy(m_cudaLeftMapData, m_leftMapData, sizeof(float)*m_outputImageSize * 2, cudaMemcpyHostToDevice);
			err = cudaMemcpy(m_cudaRightMapData, m_rightMapData, sizeof(float)*m_outputImageSize * 2, cudaMemcpyHostToDevice);

			float alpha, lamda, gamma, a;
			for (unsigned int i = 0; i < m_blendWidth * 2; i++)
			{
				alpha = static_cast<float>(i) / (m_blendWidth * 2);
				lamda = 5.2f, gamma = 1.0f, a = 0.5f;
				if (alpha <= 0.5)
				{
					alpha = a*pow(2.0f*alpha, lamda);
				}
				else
				{
					alpha = 1.0f - (1.0f - a)*pow(2.0f*(1.0f - alpha), lamda);
				}
				alpha = pow(alpha, gamma);
				m_alphaTable.push_back(alpha);
			}

			err = cudaMalloc(&m_cudaAlphaTable, sizeof(float)*m_alphaTable.size());
			err = cudaMemcpy(m_cudaAlphaTable, m_alphaTable.data(), sizeof(float)*m_alphaTable.size(), cudaMemcpyHostToDevice);
			m_alphaTable.clear();
		}
		else if (m_blenderType == 2)
		{

		}
		else
		{

		}

		err = cudaMalloc(&m_cudaOutputBuffer, m_outputImageSize * m_channels * sizeof(unsigned char));

		m_numBlocksBlend.x = m_outputWidth / m_threadsPerBlock.x;
		m_numBlocksBlend.y = m_outputHeight / m_threadsPerBlock.y;
		m_numBlocksBlend.z = 1;

		cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
		err = cudaMallocArray(&m_cudaArray, &channel_desc, m_inputWidth, m_inputHeight);
	}
}

void CCUDABlender::runBlender(unsigned char* input_data, unsigned char* output_data)
{
	cudaError_t err = cudaSuccess;

	err = cudaMemcpyToArray(m_cudaArray, 0, 0, input_data, m_inputImageSize * m_channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
	err = cuFinishToBlender(m_cudaArray, m_cudaLeftMapData, m_cudaRightMapData, m_cudaAlphaTable, m_outputWidth, m_outputHeight, m_blendWidth, m_threadsPerBlock, m_numBlocksBlend, m_cudaOutputBuffer, m_blenderType);
	err = cudaMemcpy(output_data, m_cudaOutputBuffer, m_outputImageSize * m_channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		LOGERR("Error ocurred while blending...Code: %d", err);
	}
}

void CCUDABlender::destroyBlender()
{
	if (nullptr != m_cudaLeftMapData)
	{
		cudaFree(m_cudaLeftMapData);
		m_cudaLeftMapData = nullptr;
	}

	if (nullptr != m_cudaRightMapData)
	{
		cudaFree(m_cudaRightMapData);
		m_cudaRightMapData = nullptr;
	}

	if (nullptr != m_cudaOutputBuffer)
	{
		cudaFree(m_cudaOutputBuffer);
		m_cudaOutputBuffer = nullptr;
	}

	if (nullptr != m_cudaArray)
	{
		cudaFreeArray(m_cudaArray);
		m_cudaArray = nullptr;
	}

	if (nullptr != m_unrollMap)
	{
		delete m_unrollMap;
		m_unrollMap = nullptr;
	}
}

bool CCUDABlender::setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned oh, std::string offset, int type)
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
		m_blendWidth = 5 * m_inputWidth / 360;
		// To indicate the parameters have changed.
		m_paramsChanged = true;
	}

	return true;
}

bool CCUDABlender::initializeDevice()
{
	cudaError_t err = cudaSuccess;
	int count = 0;
	err = cudaGetDeviceCount(&count);
	if (cudaSuccess != err)
	{
		LOGERR("Error Desc: %s, Error Code: %d", cudaGetErrorString(err), err);
		return false;
	}

	if (0 == count)
	{
		LOGERR("Could NOT find available CUDA device! Device count: %d", count);
		return false;
	}

	int index;
	cudaDeviceProp props;
	for (index = 0; index < count; index++)
	{
		if (cudaSuccess == cudaGetDeviceProperties(&props, index))
		{
			// 只要找到一个Compute Capability >= 1的设备就行
			if (props.major >= 1)
			{
				break;
			}
		}
	}

	if (index == count)
	{
		LOGERR("Could NOT find CUDA device with compute capability greater than 1.x");
		return false;
	}

	// Select GPU device
	err = cudaSetDevice(index);
	if (cudaSuccess != err)
	{
		LOGERR("Error Desc: %s, Error Code: %d", cudaGetErrorString(err), err);
		return false;
	}

	return true;
}
