#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cmath> 
#include <vector>
#include <chrono>
#include <memory>
#include <fstream> 
#include <time.h>  
#include "../utils/log.h"

texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> tex;

// kernel function for 3d blender 
__global__ void threeDBlender(int image_width, int image_height, float *left_map, float *right_map, unsigned char *out_img)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = y * image_width + x;
	int pivot = image_width / 2;  
	float* location = nullptr;

	if (x < pivot)
	{
		location = left_map + (index )* 2;
	}
	else
	{
		location = left_map + (index ) * 2;
	}
	// 左右半圆形设置为黑色
	int tempX = (int)location[0];
	int tempY = (int)location[1];
	if (tempX == -1 && tempY == -1)
	{
		out_img[4 * index + 0] = 0;
		out_img[4 * index + 1] = 0;
		out_img[4 * index + 2] = 0;
		out_img[4 * index + 3] = 255;
	}
	else
	{
		float4 val = tex2D(tex, location[0], location[1]);

		out_img[4 * index + 0] = val.x * 255;
		out_img[4 * index + 1] = val.y * 255;
		out_img[4 * index + 2] = val.z * 255;
		out_img[4 * index + 3] = 255;
	}
}

// kernel function for panoramic blender
__global__ void panoramicBlender(int blend_width, int image_width, int image_height, float *left_map, float *right_map, float *alpha_table, unsigned char *out_img)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = (y * image_width + x)>>1; 
	int left_start = image_width >> 2;
	float *location = nullptr;
	//  _______________________________________________________
	// |             | | |                  | | |              |
	// |             | | |                  | | |              |
	// |             | | |                  | | |              |
	// |	 MAP     | | |        MAP       | | |     MAP      |
	// |             | | |                  | | |              |
	// |             | | |                  | | |              |
	// |_____________|_|_|__________________|_|_|______________|
	//                1/4                    3/4
	//            BLENDER_LEFT          BLENDER_RIGHT
	if (x < left_start - blend_width)
	{
		location = left_map + index * 2;
		goto MAP;
	}
	else if (x >= left_start - blend_width && x < left_start + blend_width)
	{
		goto BLEND_LEFT;
	}
	else if (x >= left_start + blend_width && x < left_start * 3 - blend_width)
	{
		location = right_map + index * 2;
		goto MAP;
	}
	else if (x >= left_start * 3 - blend_width && x < left_start * 3 + blend_width)
	{
		goto BLEND_RIGHT;
	}
	else if (x >= left_start * 3 + blend_width)
	{
		location = left_map + index * 2;
		goto MAP;
	}

MAP:
	{
		float4 val = tex2D(tex, location[0], location[1]);
		out_img[4 * index + 0] = val.x * 255;
		out_img[4 * index + 1] = val.y * 255;
		out_img[4 * index + 2] = val.z * 255;
		out_img[4 * index + 3] = 255;
		return;
	}

BLEND_LEFT:
	{
		float *location1 = nullptr;
		int plane_left = left_start - blend_width;
		float alpha = *(alpha_table + 2 * blend_width - (x - plane_left) - 1);

		location = left_map + index * 2;
		location1 = right_map + index * 2;
		float4 val0 = tex2D(tex, location[0], location[1]);
		float4 val1 = tex2D(tex, location1[0], location1[1]);

		out_img[4 * index + 0] = (val0.x*alpha + val1.x*(1 - alpha)) * 255;
		out_img[4 * index + 1] = (val0.y*alpha + val1.y*(1 - alpha)) * 255;
		out_img[4 * index + 2] = (val0.z*alpha + val1.z*(1 - alpha)) * 255;
		out_img[4 * index + 3] = 255;
		return;
	}

BLEND_RIGHT:
	{
		float *location1 = nullptr;
		int plane_left = left_start * 3 - blend_width;
		float alpha = *(alpha_table + (x - plane_left));

		location = left_map + index * 2;
		location1 = right_map + index * 2;
		float4 val0 = tex2D(tex, location[0], location[1]);
		float4 val1 = tex2D(tex, location1[0], location1[1]);

		out_img[4 * index + 0] = (val0.x*alpha + val1.x*(1 - alpha)) * 255;
		out_img[4 * index + 1] = (val0.y*alpha + val1.y*(1 - alpha)) * 255;
		out_img[4 * index + 2] = (val0.z*alpha + val1.z*(1 - alpha)) * 255;
		out_img[4 * index + 3] = 255;
		return;
	}


}
//cuda  blender

extern "C" cudaError_t fishEyeBlender(cudaArray *inputBuffer, float *left_map, float*right_map, float* alpha_table, int image_width, int image_height, int bd_width, dim3 thread, dim3 numBlock, unsigned char *uOutBuffer, int type, bool paramsChanged)
{
	cudaError_t ret = cudaSuccess;

	if (paramsChanged)
	{
		tex.addressMode[0] = cudaAddressModeClamp;
		tex.addressMode[1] = cudaAddressModeClamp;
		tex.normalized = false;
		tex.filterMode = cudaFilterModeLinear;
		ret = cudaBindTextureToArray(tex, inputBuffer);
		if (cudaSuccess != ret)
		{
			LOGERR("Description:%s, Error Code: %d", cudaGetErrorString(ret), ret);
			return ret;
		}
	}

	if (type == 1)
	{
		panoramicBlender << <numBlock, thread >> >(bd_width, image_width, image_height, left_map, right_map, alpha_table, uOutBuffer);
	}
	else
	{
		threeDBlender << <numBlock, thread >> >(image_width, image_height, left_map, right_map, uOutBuffer);
	}
	ret = cudaDeviceSynchronize();
	if (cudaSuccess != ret)
	{
		LOGERR("Description:%s, Error Code: %d", cudaGetErrorString(ret), ret);
		return ret;
	}

	return ret;
} 