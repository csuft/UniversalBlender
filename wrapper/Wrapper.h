#ifndef C_WRAPPER_H__
#define C_WRAPPER_H__  

#ifdef _WINDOWS

#define EXPORTS_API _declspec( dllexport )

#ifdef __cplusplus  
extern "C" {
#endif  

	EXPORTS_API void initializeBlender(void);
	EXPORTS_API void runBlender(unsigned int input_width, unsigned int input_height, unsigned int output_width, unsigned int output_height, unsigned char* input_data, unsigned char* output_data, char* offset); 

#ifdef __cplusplus  
};
#endif  

#endif

#endif



