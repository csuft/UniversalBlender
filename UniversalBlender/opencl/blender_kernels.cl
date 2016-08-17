const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel  void  opencl_blend(__read_only    image2d_t imgSrc,		 // ‰»Î                                                                          
								__write_only   image2d_t imgDst,	 // ‰≥ˆ                                                                      
								__global float *fLeftMapBuffer,		 //left map buffer                                                           
								__global float *fRightMapBuffer,	 //right map buffer                                                          
								__global int   *_params,			 //params                                                                    
								__const float fwidthFactor,			 //width  factor                                                              
								__const float fheightFactor)		 //height factor                                                              
{
	int gidx = get_global_id(0);
	int gidy = get_global_id(1);
	int2  coord = (int2)(gidx, gidy);
	int   blendr_start1 = _params[0];
	int   blendr_end1 = _params[1];
	int   blendr_start2 = _params[2];
	int   blendr_end2 = _params[3];
	int   map_width_step = _params[4];
	int   blender_width = blendr_end1 - blendr_start1;
	int   pointer = gidy * map_width_step + 2 * gidx;
	float ratio;
	if (gidx < blendr_start1 || gidx > blendr_end2)   //  1/4 * width - blend_width/2                                                           
	{
		int  LeftU = fLeftMapBuffer[pointer];
		int  LeftV = fLeftMapBuffer[pointer + 1];
		float2 coordinate = (float2)(LeftU, LeftV);
		float2  normalizedCoordinate = convert_float2(coordinate) *(float2)(fwidthFactor, fheightFactor);
		float4  colour = read_imagef(imgSrc, sampler, normalizedCoordinate);
		write_imagef(imgDst, coord, colour);
		return;
	}
	else if (gidx > blendr_end1 && gidx < blendr_start2)
	{
		int  RightU = (int)(fRightMapBuffer[pointer]);
		int  RightV = (int)(fRightMapBuffer[pointer + 1]);
		float2 coordinate = (float2)(RightU, RightV);
		float2  normalizedCoordinate = convert_float2(coordinate) *(float2)(fwidthFactor, fheightFactor);
		float4  colour = read_imagef(imgSrc, sampler, normalizedCoordinate);
		write_imagef(imgDst, coord, colour);
		return;
	}
	else if (gidx >= blendr_start1 && gidx <= blendr_end1)
	{
		ratio = (gidx - blendr_start1) * 1.0 / blender_width;
		int  LeftU = fLeftMapBuffer[pointer];
		int  LeftV = fLeftMapBuffer[pointer + 1];
		float2 coordLeft = (float2)(LeftU, LeftV);
		float2  normCoordLeft = convert_float2(coordLeft) *(float2)(fwidthFactor, fheightFactor);
		float4  colourLeft = read_imagef(imgSrc, sampler, normCoordLeft);
		int  RightU = (int)(fRightMapBuffer[pointer]);
		int  RightV = (int)(fRightMapBuffer[pointer + 1]);
		float2 coorRight = (float2)(RightU, RightV);
		float2  normCoordRight = convert_float2(coorRight) *(float2)(fwidthFactor, fheightFactor);
		float4  colourRight = read_imagef(imgSrc, sampler, normCoordRight);
		float4 newPixel = colourRight*(float4)(ratio, ratio, ratio, 1.0) + colourLeft*(float4)((1 - ratio), (1 - ratio), (1 - ratio), 1.0);
		write_imagef(imgDst, coord, newPixel);
		return;
	}
	else if (gidx >= blendr_start2 && gidx <= blendr_end2)
	{
		ratio = 1.0 - (gidx - blendr_start2) * 1.0 / blender_width;
		int  LeftU = fLeftMapBuffer[pointer];
		int  LeftV = fLeftMapBuffer[pointer + 1];
		float2 coordLeft = (float2)(LeftU, LeftV);
		float2  normCoordLeft = convert_float2(coordLeft) *(float2)(fwidthFactor, fheightFactor);
		float4  colourLeft = read_imagef(imgSrc, sampler, normCoordLeft);
		int  RightU = (int)(fRightMapBuffer[pointer]);
		int  RightV = (int)(fRightMapBuffer[pointer + 1]);
		float2 coorRight = (float2)(RightU, RightV);
		float2  normCoordRight = convert_float2(coorRight) *(float2)(fwidthFactor, fheightFactor);
		float4  colourRight = read_imagef(imgSrc, sampler, normCoordRight);
		float4  newPixel = colourRight*(float4)(ratio, ratio, ratio, 1.0) + colourLeft*(float4)((1 - ratio), (1 - ratio), (1 - ratio), 1.0);
		write_imagef(imgDst, coord, newPixel);
		return;
	}
}

// Convert RGB(BGR) to RGBA(BGRA) 
__kernel void add_alpha_channel(__global unsigned char* input, __global unsigned char* output)
{


}

// Convert RGBA(BGRA) to RGB(BGR)
__kernel void remove_alpha_channel(__global unsigned char* input, __global unsigned char* output)
{


}
