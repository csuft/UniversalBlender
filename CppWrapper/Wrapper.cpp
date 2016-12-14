#include "Wrapper.h"
#include "BlenderWrapper.h"

#ifdef __cplusplus  
extern "C" {
#endif  
	CBlenderWrapper blender;

	void initializeBlender(void)
	{
		blender.capabilityAssessment();
		blender.getSingleInstance(CBlenderWrapper::FOUR_CHANNELS);
		blender.initializeDevice();
	}

	void runBlender(unsigned int input_width, unsigned int input_height, unsigned int output_width, unsigned int output_height, unsigned char* input_data, unsigned char* output_data, char* offset)
	{
		BlenderParams params;
		params.input_width = input_width;
		params.input_height = input_height;
		params.output_width = output_width;
		params.output_height = output_height;
		params.offset = offset;
		params.input_data = input_data;
		params.output_data = output_data;
		blender.runImageBlender(params, CBlenderWrapper::PANORAMIC_BLENDER);
	}

#ifdef __cplusplus  
};
#endif

