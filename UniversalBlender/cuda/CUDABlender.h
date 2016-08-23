#ifndef CUDA_BLENDER_H
#define CUDA_BLENDER_H

#include "../base/BaseBlender.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_texture_types.h>
#include <vector>

class CCUDABlender : public CBaseBlender
{
public:
	CCUDABlender();
	~CCUDABlender();

	virtual void setupBlender();
	virtual void runBlender(unsigned char* input_data, unsigned char* output_data);
	virtual void destroyBlender();

	// private data member
private:
	unsigned int m_inputImageSize;
	unsigned int m_outputImageSize;
	dim3 m_threadsPerBlock;
	dim3 m_numBlocksBlend;
	unsigned char* m_cudaInputBuffer;
	unsigned char* m_cudaOutputBuffer;
	cudaArray* m_cudaArray;
	std::vector<float> m_alphaTable;
	float* m_cudaAlphaTable;

	float* m_cudaLeftMapData;
	float* m_cudaRightMapData;

	// private method member
private:
	bool initializeDevice();
	bool setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned oh, std::string offset, int type);
};


#endif

