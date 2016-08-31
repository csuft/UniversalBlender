#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "wrapper/BlenderWrapper.h" 

// 输入宽高
#define INPUT_WIDTH 4096
#define INPUT_HEIGHT 2048

// 输出宽高
#define OUTPUT_WIDTH 2048
#define OUTPUT_HEIGHT 1024

using namespace cv;

void RGB2RGBA(unsigned char* rgba, unsigned char* rgb, int imageSize)
{
	if (rgba == nullptr || rgb == nullptr || imageSize <= 0)
	{
		return;
	}
	int rgbIndex = 0;
	int rgbaIndex = 0;

	while (rgbIndex < imageSize) { 
		rgba[rgbaIndex] = rgb[rgbIndex];
		rgba[rgbaIndex + 1] = rgb[rgbIndex + 1];
		rgba[rgbaIndex + 2] = rgb[rgbIndex + 2];
		rgba[rgbaIndex + 3] = 255;
		rgbIndex += 3;
		rgbaIndex += 4;
	}
}

void RGBA2RGB(unsigned char* rgb, unsigned char* rgba, int imageSize)
{
	if (rgba == nullptr || rgb == nullptr || imageSize <= 0)
	{
		return;
	}

	int rgbIndex = 0;
	int rgbaIndex = 0;

	while (rgbaIndex < imageSize) {
		rgb[rgbIndex] = rgba[rgbaIndex];
		rgb[rgbIndex + 1] = rgba[rgbaIndex + 1];
		rgb[rgbIndex + 2] = rgba[rgbaIndex + 2];

		rgbIndex += 3;
		rgbaIndex += 4;
	}
}

void testCUDA()
{
	std::string offset = "2_748.791_759.200_758.990_0.000_0.000_90.000_742.211_2266.919_750.350_-0.300_0.100_90.030_3040_1520_1026";
	unsigned char* inputImage = new unsigned char[4 * INPUT_WIDTH* INPUT_HEIGHT];
	FILE* file = fopen("3d.dat", "rb");
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

	// 1. capabilityAssessment()
	// 2. getSingleInstance();
	// 3. initializeDevice();
	// 4. runImageBlender();
	CBlenderWrapper* wrapper = new CBlenderWrapper;
	wrapper->capabilityAssessment();
	wrapper->getSingleInstance(4);
	wrapper->initializeDevice();
	wrapper->runImageBlender(params, CBlenderWrapper::THREEDIMENSION_BLENDER);

	cv::imshow("Blender Result", outputImage);
	cv::imwrite("BlenderResult_CUDA_3D.jpg", outputImage);

	delete wrapper;
}

void testOpenCL()
{
	std::string offset = "2_748.791_759.200_758.990_0.000_0.000_90.000_742.211_2266.919_750.350_-0.300_0.100_90.030_3040_1520_1026";
	unsigned char* inputImage = new unsigned char[4 * INPUT_WIDTH * INPUT_HEIGHT];
	FILE* file = fopen("3d.dat", "rb");
	fread(inputImage, 1, 4*INPUT_WIDTH*INPUT_HEIGHT, file);
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

	// 1. capabilityAssessment()
	// 2. getSingleInstance();
	// 3. initializeDevice();
	// 4. runImageBlender();
	CBlenderWrapper* wrapper = new CBlenderWrapper;
	wrapper->capabilityAssessment();
	wrapper->getSingleInstance(4);
	wrapper->initializeDevice();
	wrapper->runImageBlender(params, CBlenderWrapper::THREEDIMENSION_BLENDER);

	cv::imshow("Blender Result", outputImage); 
	cv::imwrite("BlenderResult_OpenCL_3D.jpg", outputImage);

	delete wrapper;
	delete[] inputImage;
}

void testCPU()
{
	std::string offset = "2_748.791_759.200_758.990_0.000_0.000_90.000_742.211_2266.919_750.350_-0.300_0.100_90.030_3040_1520_1026";
	cv::Mat inputImage = cv::imread("3d.jpg");
	cv::Mat outputImage(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC3);

	BlenderParams params;
	params.input_width = inputImage.cols;
	params.input_height = inputImage.rows;
	params.output_width = OUTPUT_WIDTH;
	params.output_height = OUTPUT_HEIGHT;
	params.offset = offset;
	params.input_data = inputImage.data;
	params.output_data = outputImage.data;

	// 1. capabilityAssessment()
	// 2. getSingleInstance();
	// 3. initializeDevice();
	// 4. runImageBlender();
	CBlenderWrapper* wrapper = new CBlenderWrapper;
	wrapper->capabilityAssessment();
	wrapper->getSingleInstance(3);
	wrapper->initializeDevice();
	wrapper->runImageBlender(params, CBlenderWrapper::THREEDIMENSION_BLENDER);

	cv::imshow("Blender Result", outputImage); 
	cv::imwrite("BlenderResult_CPU_3D.jpg", outputImage);

	delete wrapper;
}

void test_RGBA2RGB()
{
	unsigned char* inputImage = new unsigned char[4 * INPUT_WIDTH * INPUT_HEIGHT];
	unsigned char* outputImage = new unsigned char[3 * INPUT_WIDTH * INPUT_HEIGHT];
	FILE* infile = fopen("original_rgba.dat", "rb");
	fread(inputImage, 1, 4 * INPUT_WIDTH*INPUT_HEIGHT, infile);
	fclose(infile);

	RGBA2RGB(outputImage, inputImage, 4*INPUT_WIDTH*INPUT_HEIGHT);

	FILE* outfile = fopen("transformed_rgb.dat", "wb");
	fwrite(outputImage, 1, 3 * INPUT_WIDTH*INPUT_HEIGHT, outfile);
	fclose(outfile);
	 
}

void test_RGB2RGBA()
{
	unsigned char* inputImage = new unsigned char[3 * INPUT_WIDTH * INPUT_HEIGHT];
	unsigned char* outputImage = new unsigned char[4 * INPUT_WIDTH * INPUT_HEIGHT];
	FILE* infile = fopen("transformed_rgb.dat", "rb");
	fread(inputImage, 1, 3 * INPUT_WIDTH*INPUT_HEIGHT, infile);
	fclose(infile);

	RGB2RGBA(outputImage, inputImage, 3 * INPUT_WIDTH*INPUT_HEIGHT);

	FILE* outfile = fopen("transformed_rgba.dat", "wb");
	fwrite(outputImage, 1, 4 * INPUT_WIDTH*INPUT_HEIGHT, outfile);
	fclose(outfile);

}

void convert(const char* path)
{
	cvNamedWindow("imagetest1", CV_WINDOW_AUTOSIZE);
	IplImage* cvimg = 0;
	Mat tempOutImage(2048, 4096, CV_8UC4);
	IplImage* opencvImage = new IplImage(tempOutImage);
	unsigned char rgbTmp[3];
	cvimg = cvLoadImage(path, CV_LOAD_IMAGE_COLOR);

	cvShowImage("imagetest1", cvimg);
	for (int y = 0; y < cvimg->height; y++){
		for (int x = 0; x < cvimg->width; x++) {
			rgbTmp[0] = cvimg->imageData[cvimg->widthStep * y + x * 3];					 // B  
			rgbTmp[1] = cvimg->imageData[cvimg->widthStep * y + x * 3 + 1];				 // G  
			rgbTmp[2] = cvimg->imageData[cvimg->widthStep * y + x * 3 + 2];				 // R   

			opencvImage->imageData[opencvImage->widthStep * y + x * 4] = rgbTmp[2];      // R
			opencvImage->imageData[opencvImage->widthStep * y + x * 4 + 1] = rgbTmp[1];  // G
			opencvImage->imageData[opencvImage->widthStep * y + x * 4 + 2] = rgbTmp[0];  // B
			opencvImage->imageData[opencvImage->widthStep * y + x * 4 + 3] = 255;        // A
		}
	}
	unsigned char *dataBackGround = (unsigned char *)malloc(sizeof(unsigned char)*opencvImage->imageSize);
	memcpy(dataBackGround, opencvImage->imageData, opencvImage->imageSize);

	FILE* file = fopen("3d.dat", "wb");
	fwrite(dataBackGround, 1, opencvImage->imageSize, file);
	fclose(file);

	Mat outputImage(2048, 4096, CV_8UC4);
	outputImage.data = dataBackGround;
	imwrite("output.png", outputImage);
}

int main(void)
{
	//convert("3d.jpg");
	//testCPU();
	//testOpenCL();
	//testCUDA();
	//test_RGBA2RGB();
	test_RGB2RGBA();

	return 0;
}
