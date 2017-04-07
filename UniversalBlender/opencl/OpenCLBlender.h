#ifndef OPENCL_BLENDER_H
#define OPENCL_BLENDER_H

#include "../base/BaseBlender.h"

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#include <OpenCL/cl_gl.h> 
#else  
#include <CL/cl.hpp>
#include <CL/cl_gl.h>
#endif 

class COpenCLBlender : public CBaseBlender
{
public:
	COpenCLBlender();
	COpenCLBlender(int mode);
	~COpenCLBlender();

	bool initializeDevice();
	virtual void setupBlender();
	virtual void runBlender(unsigned char* input_data, unsigned char* output_data);
	virtual void destroyBlender();
	virtual bool setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned int oh, std::string offset, int type);

	//private data member
private:
	inline bool checkError(cl_int err, const char* name);
	const char* get_error_string(cl_int err);
	int findNearestNumber(unsigned int max);
	// private method member
private:
	unsigned int m_inputImageSize;
	unsigned int m_outputImageSize; 
	cl::size_t<3> m_origins;
	cl::size_t<3> m_inputRegions;
	cl::size_t<3> m_outputRegions;
	float m_widthFactor, m_heightFactor;
	int m_inputParams[16];
	cl::Context* m_openclContext;
	cl::Program* m_program;

	unsigned char* m_inputBuffer;
	cl::Image2D* m_inputImage;

	unsigned char* m_outputBuffer;
	cl::Image2D* m_outputImage;

	cl::Buffer* m_inputParamsBuffer;
	cl::Buffer* m_leftMapBuffer;
	cl::Buffer* m_rightMapBuffer;

	cl::CommandQueue* m_commandQueue;
	cl::Kernel* m_kernel;

	// Nearest number
	int m_nearestNum;
};

static const std::string BLEND_KERNEL_STRING =
"const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;                                               \n"
"__kernel  void  opencl_blend(   __read_only    image2d_t imgSrc,   // output image                                                                  \n"
"	                            __write_only   image2d_t imgDst,    // input iamge                                                                   \n"
"	                            __global float *fLeftMapBuffer,     // left map buffer                                                               \n"
"	                            __global float *fRightMapBuffer,    // right map buffer                                                              \n"
"	                            __global int   *_params,            // params                                                                        \n"
"	                            __const float fwidthFactor,         // width  factor                                                                 \n"
"	                            __const float fheightFactor,        // height factor                                                                 \n"
"                               __const int type)                   // blender type                                                                  \n"
"{                                                                                                                                                   \n"
"	int gidx = get_global_id(0);                                                                                                                     \n"
"	int gidy = get_global_id(1);                                                                                                                     \n"
"	int2  coord = (int2)(gidx, gidy);                                                                                                                \n"
"   if (type == 1 || type == 3) // Panoramic Blender is the same as panoramic cylinder blender mode                                                  \n"
"	{                                                                                                                                                \n"
"		int   blendr_start1  = _params[0];                                                                                                           \n"
"		int   blendr_end1    = _params[1];                                                                                                           \n"
"		int   blendr_start2  = _params[2];                                                                                                           \n"
"		int   blendr_end2    = _params[3];                                                                                                           \n"
"		int   map_width_step = _params[4];                                                                                                           \n"
"		int   blender_width  = blendr_end1 - blendr_start1;                                                                                          \n"
"		int   pointer        = gidy * map_width_step + 2 * gidx;                                                                                     \n"
"		float ratio;                                                                                                                                 \n"
"		if (gidx < blendr_start1 || gidx > blendr_end2)   //  1/4 * width - blend_width/2                                                            \n"
"		{                                                                                                                                            \n"
"			int  LeftU = fLeftMapBuffer[pointer];                                                                                                    \n"
"			int  LeftV = fLeftMapBuffer[pointer + 1];                                                                                                \n"
"			float2 coordinate = (float2)(LeftU, LeftV);                                                                                              \n"
"			float2  normalizedCoordinate = convert_float2(coordinate) *(float2)(fwidthFactor, fheightFactor);                                        \n"
"			float4  colour = read_imagef(imgSrc, sampler, normalizedCoordinate);                                                                     \n"
"			write_imagef(imgDst, coord, colour);                                                                                                     \n"
"		}                                                                                                                                            \n"
"		else if (gidx > blendr_end1 && gidx < blendr_start2)                                                                                         \n"
"		{                                                                                                                                            \n"
"			int  RightU = (int)(fRightMapBuffer[pointer]);                                                                                           \n"
"			int  RightV = (int)(fRightMapBuffer[pointer + 1]);                                                                                       \n"
"			float2 coordinate = (float2)(RightU, RightV);                                                                                            \n"
"			float2  normalizedCoordinate = convert_float2(coordinate) *(float2)(fwidthFactor, fheightFactor);                                        \n"
"			float4  colour = read_imagef(imgSrc, sampler, normalizedCoordinate);                                                                     \n"
"			write_imagef(imgDst, coord, colour);                                                                                                     \n"
"		}                                                                                                                                            \n"
"		else if (gidx >= blendr_start1 && gidx <= blendr_end1)                                                                                       \n"
"		{                                                                                                                                            \n"
"			ratio = (gidx - blendr_start1) * 1.0 / blender_width;                                                                                    \n"
"			int  LeftU = fLeftMapBuffer[pointer];                                                                                                    \n"
"			int  LeftV = fLeftMapBuffer[pointer + 1];                                                                                                \n"
"			float2 coordLeft = (float2)(LeftU, LeftV);                                                                                               \n"
"			float2  normCoordLeft = convert_float2(coordLeft) *(float2)(fwidthFactor, fheightFactor);                                                \n"
"			float4  colourLeft = read_imagef(imgSrc, sampler, normCoordLeft);                                                                        \n"
"			int  RightU = (int)(fRightMapBuffer[pointer]);                                                                                           \n"
"			int  RightV = (int)(fRightMapBuffer[pointer + 1]);                                                                                       \n"
"			float2 coorRight = (float2)(RightU, RightV);                                                                                             \n"
"			float2  normCoordRight = convert_float2(coorRight) *(float2)(fwidthFactor, fheightFactor);                                               \n"
"			float4  colourRight = read_imagef(imgSrc, sampler, normCoordRight);                                                                      \n"
"			float4 newPixel = colourRight*(float4)(ratio, ratio, ratio, 1.0) + colourLeft*(float4)((1 - ratio), (1 - ratio), (1 - ratio), 1.0) ;     \n"
"			write_imagef(imgDst, coord, newPixel);                                                                                                   \n"
"		}                                                                                                                                            \n"
"		else if (gidx >= blendr_start2 && gidx <= blendr_end2)                                                                                       \n"
"		{                                                                                                                                            \n"
"			ratio = 1.0 - (gidx - blendr_start2) * 1.0 / blender_width;                                                                              \n"
"			int  LeftU = fLeftMapBuffer[pointer];                                                                                                    \n"
"			int  LeftV = fLeftMapBuffer[pointer + 1];                                                                                                \n"
"			float2 coordLeft = (float2)(LeftU, LeftV);                                                                                               \n"
"			float2  normCoordLeft = convert_float2(coordLeft) *(float2)(fwidthFactor, fheightFactor);                                                \n"
"			float4  colourLeft = read_imagef(imgSrc, sampler, normCoordLeft);                                                                        \n"
"			int  RightU = (int)(fRightMapBuffer[pointer]);                                                                                           \n"
"			int  RightV = (int)(fRightMapBuffer[pointer + 1]);                                                                                       \n"
"			float2 coorRight = (float2)(RightU, RightV);                                                                                             \n"
"			float2  normCoordRight = convert_float2(coorRight) *(float2)(fwidthFactor, fheightFactor);                                               \n"
"			float4  colourRight = read_imagef(imgSrc, sampler, normCoordRight);                                                                      \n"
"			float4  newPixel = colourRight*(float4)(ratio, ratio, ratio, 1.0) + colourLeft*(float4)((1 - ratio), (1 - ratio), (1 - ratio), 1.0);     \n"
"			write_imagef(imgDst, coord, newPixel);                                                                                                   \n"
"		}                                                                                                                                            \n"
"	}	                                                                                                                                             \n"
"   else if (type == 2) // 3D Blender                                                                                                                \n"
"	{                                                                                                                                                \n"
"		int centerWidth  = _params[4] / 2;                                                                                                         \n"
"		float4 colour = (float4)(0, 0, 0, 255);                                                                                                      \n"
"		int	pointer = gidy * centerWidth * 2 + 2 * (gidx);                                                                                           \n"
"		int  coordU = fLeftMapBuffer[pointer];                                                                                                       \n"
"		int  coordV = fLeftMapBuffer[pointer + 1];                                                                                                   \n"
"		if (coordU != -1 && coordV != -1)                                                                                                            \n"
"		{                                                                                                                                            \n"
"			float2 coordinate = (float2)(coordU, coordV);                                                                                            \n"
"			float2  normalizedCoordinate = convert_float2(coordinate) *(float2)(fwidthFactor, fheightFactor);                                        \n"
"			colour = read_imagef(imgSrc, sampler, normalizedCoordinate);                                                                             \n"
"		}                                                                                                                                            \n"
"                                                                                                                                                    \n"
"		write_imagef(imgDst, coord, colour);                                                                                                         \n"
"	}                                                                                                                                                \n"
"	else                                                                                                                                             \n"
"	{                                                                                                                                                \n"
"       int centerHeight    = _params[5] / 2;                                                                                                        \n"
"       int width = _params[4]/2;                                                                                                                    \n"
"       float4 colour = (float4)(0, 0, 0, 255);                                                                 								     \n"
"       int	pointer, coordU, coordV;                                                                            								     \n"
"       pointer = gidy * width * 2 + 2 * (gidx);                                                         								             \n"
"	    coordU = fLeftMapBuffer[pointer];                                                                    								         \n"
"       coordV = fLeftMapBuffer[pointer + 1];                                                                								         \n"
"       if (coordU != -1 && coordV != -1)                                                                       								     \n"
"       {                                                                                                       								     \n"
"	       float2 coordinate = (float2)(coordU, coordV);                                                        								     \n"
"	       float2  normalizedCoordinate = convert_float2(coordinate) *(float2)(fwidthFactor, fheightFactor);    								     \n"
"	       colour = read_imagef(imgSrc, sampler, normalizedCoordinate);                                         								     \n"
"       }                                                                                                       								     \n"
"                                                                                                               								     \n"
"       write_imagef(imgDst, coord, colour);                                                                    								     \n"
"	}                                                                                                                                                \n"
"	                                                                                                                                                 \n"
"}                                                                                                                                                   \n"

;


#endif


