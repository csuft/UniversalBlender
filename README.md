Universal Blender
=================
This project is created to implement a universal blender for panorama images and videos by taking the advantage of parallel computing technologies(CUDA/OpenCL). The blender prefers to use CUDA rather than CPU or OpenCL. Then it will try to use OpenCL if there is no CUDA device to use. 

Advantages
----------
* Global single instance pattern.
* Support RGB(RGBA), BGR(BGRA) color model
* Use parallel computing underlying if possible
* Support arbitrary input resolution and output resolution
* Utilize log information to help debug
* Offset can be replaced in any frequency. 
* Good inheritance hierarchy

How to compile?
---------------
* Configure OpenCL(to be continue)
* Configure CUDA(to be continue)
* Configure OpenCV(to be continue)

How to use?
-----------
The UniversalBlender can be used in any circumstances. The following code samples demonstrate how can we integrate UniversalBlender with our existing projects:  

    #include "BlenderWrapper.h"

	#define OUTPUT_WIDTH 4096
	#define OUTPUT_HEIGHT 2048
	#define INPUT_WIDTH 3040
	#define INPUT_HEIGHT 1520

	std::string offset = "2_748.791_759.200_758.990_0.000_0.000_90.000_742.211_2266.919_750.350_-0.300_0.100_90.030_3040_1520_1026";
	unsigned char* inputImage = new unsigned char[4 * INPUT_WIDTH* INPUT_HEIGHT];
	FILE* file = fopen("output.dat", "rb");
	fread(inputImage, 1, 4 * INPUT_WIDTH * INPUT_HEIGHT, file);
	fclose(file);
	cv::Mat outputImage(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC4);
	
	BlenderParams params;
	params.input_width = INPUT_WIDTH;
	params.input_height = INPUT_HEIGHT;
	params.output_width = OUTPUT_WIDTH;
	params.output_height = OUTPUT_HEIGHT;
	params.offset = offset;
	params.input_data = inputImage;
	params.output_data = outputImage.data;

	CBlenderWrapper* wrapper = new CBlenderWrapper;
	wrapper->capabilityAssessment();
	// Yeah, images must be 4 channels for OpenCL and CUDA blender.
	wrapper->getSingleInstance(4);
	wrapper->initializeDevice();
	wrapper->runImageBlender(params, CBlenderWrapper::PANORAMIC_BLENDER);  

	cv::imshow("Blender Result", outputImage);
	cv::imwrite("BlenderResult_CUDA.jpg", outputImage);

	delete wrapper;
License
-------
All copyrights reserved. Arashi Vision Ltd. 2016. 
