#ifndef CUDA_BLENDER_H
#define CUDA_BLENDER_H

#include "../base/BaseBlender.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_texture_types.h>
#include <vector>
#include <cmath>

class CCUDABlender : public CBaseBlender
{
public:
	CCUDABlender();
	CCUDABlender(int mode);
	~CCUDABlender();

	bool initializeDevice();

	virtual void setupBlender();
	virtual void runBlender(unsigned char* input_data, unsigned char* output_data);
	virtual void destroyBlender();
	virtual bool setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned oh, std::string offset, int type);
	// private data member
private:
	unsigned int m_inputImageSize;
	unsigned int m_outputImageSize;
	dim3 m_threadsPerBlock;
	dim3 m_numBlocksBlend;
	unsigned char* m_cudaOutputBuffer;
	cudaArray* m_cudaArray;
	std::vector<float> m_alphaTable;
	float* m_cudaAlphaTable;

	float* m_cudaLeftMapData;
	float* m_cudaRightMapData; 
};


#endif

