#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "wrapper/BlenderWrapper.h" 
#define OUTPUT_WIDTH 4096
#define OUTPUT_HEIGHT 2048

using namespace cv;


void testCUDA()
{
	std::string offset = "2_748.791_759.200_758.990_0.000_0.000_90.000_742.211_2266.919_750.350_-0.300_0.100_90.030_3040_1520_1026";
	cv::Mat inputImage = cv::imread("input.jpg");
	cv::Mat outputImage(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC3);

	BlenderParams params;
	params.input_width = inputImage.cols;
	params.input_height = inputImage.rows;
	params.output_width = inputImage.cols;
	params.output_height = inputImage.rows;
	params.offset = offset;
	params.input_data = inputImage.data;
	params.output_data = outputImage.data;

	// 1. capabilityAssessment()
	// 2. getSingleInstance();
	// 3. initializeDevice();
	// 4. runImageBlender();
	CBlenderWrapper* wrapper = new CBlenderWrapper;
	wrapper->capabilityAssessment();
	wrapper->getSingleInstance();
	wrapper->initializeDevice();
	wrapper->runImageBlender(params, CBlenderWrapper::PANORAMIC_BLENDER);

	cv::imshow("Blender Result", outputImage);
	cv::waitKey(20);
	cv::imwrite("BlenderResult_CUDA.jpg", outputImage);

	delete wrapper;
}

void testOpenCL()
{
	std::string offset = "2_748.791_759.200_758.990_0.000_0.000_90.000_742.211_2266.919_750.350_-0.300_0.100_90.030_3040_1520_1026";
	unsigned char* inputImage = new unsigned char[4 * 3040 * 1520];
	FILE* file = fopen("output.dat", "rb");
	fread(inputImage, 1, 4*1520*3040, file);
	fclose(file);
	cv::Mat outputImage(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC4);

	BlenderParams params;
	params.input_width = 3040;
	params.input_height = 1520;
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
	wrapper->runImageBlender(params, CBlenderWrapper::PANORAMIC_BLENDER);

	cv::imshow("Blender Result", outputImage);
	cv::waitKey(20);
	cv::imwrite("BlenderResult_OpenCL.jpg", outputImage);

	delete wrapper;
	delete[] inputImage;
}

void testCPU()
{
	std::string offset = "2_748.791_759.200_758.990_0.000_0.000_90.000_742.211_2266.919_750.350_-0.300_0.100_90.030_3040_1520_1026";
	cv::Mat inputImage = cv::imread("input.jpg");
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
	wrapper->getSingleInstance();
	wrapper->initializeDevice();
	wrapper->runImageBlender(params, CBlenderWrapper::PANORAMIC_BLENDER);

	cv::imshow("Blender Result", outputImage);
	cv::waitKey(20);
	cv::imwrite("BlenderResult_CPU.jpg", outputImage);

	delete wrapper;
}

void convert()
{
	cvNamedWindow("imagetest1", CV_WINDOW_AUTOSIZE);
	IplImage* cvimg = 0;
	Mat tempOutImage(1520, 3040, CV_8UC4);
	IplImage* opencvImage = new IplImage(tempOutImage);
	char rgbTmp[3];
	cvimg = cvLoadImage("input.jpg", CV_LOAD_IMAGE_COLOR);

	cvShowImage("imagetest1", cvimg);
	for (int y = 0; y < cvimg->height; y++){
		for (int x = 0; x < cvimg->width; x++) {
			rgbTmp[0] = cvimg->imageData[cvimg->widthStep * y + x * 3];     // B  
			rgbTmp[1] = cvimg->imageData[cvimg->widthStep * y + x * 3 + 1]; // G  
			rgbTmp[2] = cvimg->imageData[cvimg->widthStep * y + x * 3 + 2]; // R  
			//rgbTmp[0] = gray->imageData[cvimg->widthStep * y/3 + 640-x];     // B  
			//rgbTmp[1] = gray->imageData[cvimg->widthStep * y/3 + 640-x];     // G  
			//rgbTmp[2] = gray->imageData[cvimg->widthStep * y/3 + 640-x];     // R  
			opencvImage->imageData[opencvImage->widthStep * y + x * 4] = rgbTmp[0];
			opencvImage->imageData[opencvImage->widthStep * y + x * 4 + 1] = rgbTmp[1];
			opencvImage->imageData[opencvImage->widthStep * y + x * 4 + 2] = rgbTmp[2];
			opencvImage->imageData[opencvImage->widthStep * y + x * 4 + 3] = 255;
		}
	}
	unsigned char *dataBackGround = (unsigned char *)malloc(sizeof(unsigned char)*opencvImage->imageSize);
	memcpy(dataBackGround, opencvImage->imageData, opencvImage->imageSize);

	FILE* file = fopen("output.dat", "wb");
	fwrite(dataBackGround, 1, opencvImage->imageSize, file);
	fclose(file);

	Mat outputImage(1520, 3040, CV_8UC4);
	outputImage.data = dataBackGround;
	imwrite("output.png", outputImage);
}

int main(void)
{
	testOpenCL();
	return 0;
}
