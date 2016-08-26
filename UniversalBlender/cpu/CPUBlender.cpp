#include "CPUBlender.h"

#include <future>
#include <vector>
// Using delegate constructor in C++11
CCPUBlender::CCPUBlender() : CCPUBlender(4)
{

}

CCPUBlender::CCPUBlender(int channels)
{
	if (channels != 3 || channels != 4)
	{
		channels = 3;
	}
	m_channels = channels;
	m_unrollMap = new UnrollMap;
}

CCPUBlender::~CCPUBlender()
{
	destroyBlender();
}

unsigned char* CCPUBlender::addAlphaChannel(const unsigned char* inputImage)
{
	// To do lately...
	return nullptr;
}

unsigned char* CCPUBlender::removeAlphaChannel(const unsigned char* inputImage)
{
	// To do lately...
	return nullptr;
}

void CCPUBlender::runBlender(unsigned char* input_data, unsigned char* output_data)
{
	if (m_blenderType == 1)
	{
		m_blendWidth = 5 * m_outputWidth / 360;
		int m_start1, m_start2, m_end1, m_end2;
		m_start1 = (m_outputWidth >> 2) - (m_blendWidth >> 1);
		m_end1 = m_start1 + m_blendWidth;
		m_start2 = (m_outputWidth * 3 / 4) - (m_blendWidth >> 1);
		m_end2 = m_start2 + m_blendWidth;
		auto fun = [&](int yStart, int yEnd)
		{
			int rightU, rightV;
			float rightRatioX, rightRatioY;
			float rightRatio0, rightRatio1, rightRatio2, rightRatio3;

			int leftU, leftV;
			float leftRatioX, leftRatioY;
			float leftRatio0, leftRatio1, leftRatio2, leftRatio3;

			unsigned char* blendImageData;
			unsigned char* curBlendImageData;
			float* rightMapData;
			float* curRightMapData;
			unsigned char* rightImageData;
			unsigned char* rightImageDataNext;
			float* leftMapData;
			float* curLeftMapData;
			unsigned char* leftImageData;
			unsigned char* leftImageDataNext;
			float ratio;
			//右鱼眼第一部分
			for (int y = yStart; y < yEnd; y++)
			{
				blendImageData = (output_data + y * m_outputWidth * m_channels);//blendImage.ptr<uchar>(y);
				rightMapData = (m_leftMapData + y * m_outputWidth * 2);//rightMap.ptr<float>(y);
				leftMapData = (m_rightMapData + y * m_outputWidth * 2);//leftMap.ptr<float>(y);
				for (int x = 0; x < m_start1; x++)
				{
					curRightMapData = rightMapData + 2 * x;

					rightU = curRightMapData[0];
					rightV = curRightMapData[1];

					int  u = (floor)(rightU);
					int  v = (floor)(rightV);
					rightRatioX = 1 - u + rightU;
					rightRatioY = 1 - v + rightV;
					rightRatio0 = rightRatioX * rightRatioY;
					rightRatio1 = rightRatioY - rightRatio0;
					rightRatio2 = rightRatioX - rightRatio0;
					rightRatio3 = 1 - rightRatioX - rightRatioY + rightRatio0;
					rightImageData = (input_data + rightV * m_inputWidth * m_channels + m_channels * rightU);// image.ptr<uchar>(rightV) + 3 * rightU;
					rightImageDataNext = (input_data + (rightV + 1) * m_inputWidth*m_channels + m_channels * rightU);// image.ptr<uchar>(rightV + 1) + 3 * rightU;
					curBlendImageData = blendImageData + m_channels * x;
					curBlendImageData[0] = rightRatio0 * rightImageData[0] +
						rightRatio1 * rightImageData[4] +
						rightRatio2 * rightImageDataNext[0] +
						rightRatio3 * rightImageDataNext[4] + 0.5;
					curBlendImageData[1] = rightRatio0 * rightImageData[1] +
						rightRatio1 * rightImageData[5] +
						rightRatio2 * rightImageDataNext[1] +
						rightRatio3 * rightImageDataNext[5] + 0.5;
					curBlendImageData[2] = rightRatio0 * rightImageData[2] +
						rightRatio1 * rightImageData[6] +
						rightRatio2 * rightImageDataNext[2] +
						rightRatio3 * rightImageDataNext[6] + 0.5;
					if (m_channels > 3)
					{
						curBlendImageData[3] = 255;
					}
				}

				for (int x = m_start1; x < m_end1; x++)
				{
					ratio = (x - m_start1) * 1.0 / m_blendWidth;
					curLeftMapData = leftMapData + 2 * x;
					leftU = curLeftMapData[0];
					leftV = curLeftMapData[1];
					int  u = (floor)(leftU);
					int  v = (floor)(leftV);
					leftRatioX = 1 - u + leftU;
					leftRatioY = 1 - v + leftV;
					leftRatio0 = leftRatioX * leftRatioY;
					leftRatio1 = leftRatioY - leftRatio0;
					leftRatio2 = leftRatioX - leftRatio0;
					leftRatio3 = 1 - leftRatioX - leftRatioY + leftRatio0;
					leftImageData = (input_data + leftV * m_inputWidth * m_channels + m_channels * leftU);// image.ptr<uchar>(leftV) + 3 * leftU;
					leftImageDataNext = (input_data + (leftV + 1) * m_inputWidth * m_channels + m_channels * leftU);//image.ptr<uchar>(leftV + 1) + 3 * leftU;

					curRightMapData = rightMapData + 2 * x;
					rightU = curRightMapData[0];
					rightV = curRightMapData[1];
					int  ru = (floor)(rightU);
					int  rv = (floor)(rightV);
					rightRatioX = 1 - ru + rightU;
					rightRatioY = 1 - rv + rightV;
					rightRatio0 = rightRatioX * rightRatioY;
					rightRatio1 = rightRatioY - rightRatio0;
					rightRatio2 = rightRatioX - rightRatio0;
					rightRatio3 = 1 - rightRatioX - rightRatioY + rightRatio0;
					rightImageData = (input_data + rightV * m_inputWidth * m_channels + m_channels * rightU);// image.ptr<uchar>(rightV) + 3 * rightU;
					rightImageDataNext = (input_data + (rightV + 1)* m_inputWidth * m_channels + m_channels * rightU);// image.ptr<uchar>(rightV + 1) + 3 * rightU;

					float leftDataTemp0 = leftRatio0 * leftImageData[0] +
						leftRatio1 * leftImageData[4] +
						leftRatio2 * leftImageDataNext[0] +
						leftRatio3 * leftImageDataNext[4] + 0.5;
					float leftDataTemp1 = leftRatio0 * leftImageData[1] +
						leftRatio1 * leftImageData[5] +
						leftRatio2 * leftImageDataNext[1] +
						leftRatio3 * leftImageDataNext[5] + 0.5;
					float leftDataTemp2 = leftRatio0 * leftImageData[2] +
						leftRatio1 * leftImageData[6] +
						leftRatio2 * leftImageDataNext[2] +
						leftRatio3 * leftImageDataNext[6] + 0.5;

					float rightDataTemp0 = rightRatio0 * rightImageData[0] +
						rightRatio1 * rightImageData[4] +
						rightRatio2 * rightImageDataNext[0] +
						rightRatio3 * rightImageDataNext[4] + 0.5;
					float rightDataTemp1 = rightRatio0 * rightImageData[1] +
						rightRatio1 * rightImageData[5] +
						rightRatio2 * rightImageDataNext[1] +
						rightRatio3 * rightImageDataNext[5] + 0.5;
					float rightDataTemp2 = rightRatio0 * rightImageData[2] +
						rightRatio1 * rightImageData[6] +
						rightRatio2 * rightImageDataNext[2] +
						rightRatio3 * rightImageDataNext[6] + 0.5;

					curBlendImageData = blendImageData + m_channels * x;
					curBlendImageData[0] = leftDataTemp0 * ratio + rightDataTemp0 * (1 - ratio);
					curBlendImageData[1] = leftDataTemp1 * ratio + rightDataTemp1 * (1 - ratio);
					curBlendImageData[2] = leftDataTemp2 * ratio + rightDataTemp2 * (1 - ratio);
					if (m_channels > 3)
					{
						curBlendImageData[3] = 255;
					}
				}
				for (int x = m_end1; x < m_start2; x++)
				{
					curLeftMapData = leftMapData + 2 * x;
					leftU = curLeftMapData[0];
					leftV = curLeftMapData[1];
					int  u = (floor)(leftU);
					int  v = (floor)(leftV);
					leftRatioX = 1 - u + leftU;
					leftRatioY = 1 - v + leftV;
					leftRatio0 = leftRatioX * leftRatioY;
					leftRatio1 = leftRatioY - leftRatio0;
					leftRatio2 = leftRatioX - leftRatio0;
					leftRatio3 = 1 - leftRatioX - leftRatioY + leftRatio0;
					leftImageData = (input_data + leftV * m_inputWidth * m_channels + m_channels * leftU);//image.ptr<uchar>(leftV) + 3 * leftU;
					leftImageDataNext = (input_data + (leftV + 1) * m_inputWidth * m_channels + m_channels * leftU);//image.ptr<uchar>(leftV + 1) + 3 * leftU;

					curBlendImageData = blendImageData + m_channels * x;
					curBlendImageData[0] = leftRatio0 * leftImageData[0] +
						leftRatio1 * leftImageData[4] +
						leftRatio2 * leftImageDataNext[0] +
						leftRatio3 * leftImageDataNext[4] + 0.5;
					curBlendImageData[1] = leftRatio0 * leftImageData[1] +
						leftRatio1 * leftImageData[5] +
						leftRatio2 * leftImageDataNext[1] +
						leftRatio3 * leftImageDataNext[5] + 0.5;
					curBlendImageData[2] = leftRatio0 * leftImageData[2] +
						leftRatio1 * leftImageData[6] +
						leftRatio2 * leftImageDataNext[2] +
						leftRatio3 * leftImageDataNext[6] + 0.5;
					if (m_channels > 3)
					{
						curBlendImageData[3] = 255;
					}

				}
				for (int x = m_start2; x < m_end2; x++)
				{
					ratio = 1 - (x - m_start2) * 1.0 / m_blendWidth;
					curLeftMapData = leftMapData + 2 * x;
					leftU = curLeftMapData[0];
					leftV = curLeftMapData[1];
					int  u = (floor)(leftU);
					int  v = (floor)(leftV);
					leftRatioX = 1 - u + leftU;
					leftRatioY = 1 - v + leftV;
					leftRatio0 = leftRatioX * leftRatioY;
					leftRatio1 = leftRatioY - leftRatio0;
					leftRatio2 = leftRatioX - leftRatio0;
					leftRatio3 = 1 - leftRatioX - leftRatioY + leftRatio0;
					leftImageData = (input_data + leftV * m_inputWidth * m_channels + m_channels * leftU);//image.ptr<uchar>(leftV) + 3 * leftU;
					leftImageDataNext = (input_data + (leftV + 1) * m_inputWidth * m_channels + m_channels * leftU);//image.ptr<uchar>(leftV + 1) + 3 * leftU;

					curRightMapData = rightMapData + 2 * x;
					rightU = curRightMapData[0];
					rightV = curRightMapData[1];
					int  ru = (floor)(rightU);
					int  rv = (floor)(rightV);
					rightRatioX = 1 - ru + rightU;
					rightRatioY = 1 - rv + rightV;
					rightRatio0 = rightRatioX * rightRatioY;
					rightRatio1 = rightRatioY - rightRatio0;
					rightRatio2 = rightRatioX - rightRatio0;
					rightRatio3 = 1 - rightRatioX - rightRatioY + rightRatio0;
					rightImageData = (input_data + rightV *  m_inputWidth * m_channels + m_channels * rightU);//image.ptr<uchar>(rightV) + 3 * rightU;
					rightImageDataNext = (input_data + (rightV + 1) *  m_inputWidth * m_channels + m_channels * rightU);//image.ptr<uchar>(rightV + 1) + 3 * rightU;

					float leftDataTemp0 = leftRatio0 * leftImageData[0] +
						leftRatio1 * leftImageData[4] +
						leftRatio2 * leftImageDataNext[0] +
						leftRatio3 * leftImageDataNext[4] + 0.5;
					float leftDataTemp1 = leftRatio0 * leftImageData[1] +
						leftRatio1 * leftImageData[5] +
						leftRatio2 * leftImageDataNext[1] +
						leftRatio3 * leftImageDataNext[5] + 0.5;
					float leftDataTemp2 = leftRatio0 * leftImageData[2] +
						leftRatio1 * leftImageData[6] +
						leftRatio2 * leftImageDataNext[2] +
						leftRatio3 * leftImageDataNext[6] + 0.5;

					float rightDataTemp0 = rightRatio0 * rightImageData[0] +
						rightRatio1 * rightImageData[4] +
						rightRatio2 * rightImageDataNext[0] +
						rightRatio3 * rightImageDataNext[4] + 0.5;
					float rightDataTemp1 = rightRatio0 * rightImageData[1] +
						rightRatio1 * rightImageData[5] +
						rightRatio2 * rightImageDataNext[1] +
						rightRatio3 * rightImageDataNext[5] + 0.5;
					float rightDataTemp2 = rightRatio0 * rightImageData[2] +
						rightRatio1 * rightImageData[6] +
						rightRatio2 * rightImageDataNext[2] +
						rightRatio3 * rightImageDataNext[6] + 0.5;

					curBlendImageData = blendImageData + m_channels * x;
					curBlendImageData[0] = leftDataTemp0 * ratio + rightDataTemp0 * (1 - ratio);
					curBlendImageData[1] = leftDataTemp1 * ratio + rightDataTemp1 * (1 - ratio);
					curBlendImageData[2] = leftDataTemp2 * ratio + rightDataTemp2 * (1 - ratio);
					if (m_channels > 3)
					{
						curBlendImageData[3] = 255;
					}
				}

				for (int x = m_end2; x < m_outputWidth; x++)
				{
					curRightMapData = rightMapData + 2 * x;
					rightU = curRightMapData[0];
					rightV = curRightMapData[1];
					int  ru = (floor)(rightU);
					int  rv = (floor)(rightV);
					rightRatioX = 1 - ru + rightU;
					rightRatioY = 1 - rv + rightV;
					rightRatio0 = rightRatioX * rightRatioY;
					rightRatio1 = rightRatioY - rightRatio0;
					rightRatio2 = rightRatioX - rightRatio0;
					rightRatio3 = 1 - rightRatioX - rightRatioY + rightRatio0;
					rightImageData = (input_data + rightV * m_inputWidth * m_channels + m_channels * rightU);//image.ptr<uchar>(rightV) + 3 * rightU;
					rightImageDataNext = (input_data + (rightV + 1) * m_inputWidth * m_channels + m_channels * rightU);//image.ptr<uchar>(rightV + 1) + 3 * rightU;
					curBlendImageData = blendImageData + m_channels * x;
					curBlendImageData[0] = rightRatio0 * rightImageData[0] +
						rightRatio1 * rightImageData[4] +
						rightRatio2 * rightImageDataNext[0] +
						rightRatio3 * rightImageDataNext[4] + 0.5;
					curBlendImageData[1] = rightRatio0 * rightImageData[1] +
						rightRatio1 * rightImageData[5] +
						rightRatio2 * rightImageDataNext[1] +
						rightRatio3 * rightImageDataNext[5] + 0.5;
					curBlendImageData[2] = rightRatio0 * rightImageData[2] +
						rightRatio1 * rightImageData[6] +
						rightRatio2 * rightImageDataNext[2] +
						rightRatio3 * rightImageDataNext[6] + 0.5;
					if (m_channels > 3)
					{
						curBlendImageData[3] = 255;
					}
				}
			}
		};
		int threadNum = 8;
		std::vector<std::future<void>> threads(threadNum);
		int addHeight = m_outputHeight / threadNum;
		//开启多线程
		for (int i = 0; i < threadNum - 1; i++)
		{
			threads[i] = std::async(fun, i * addHeight, (i + 1) * addHeight);
		}
		threads[threadNum - 1] = std::async(fun, (threadNum - 1) * addHeight, m_outputHeight);

		//等待多线程完成
		for (int i = 0; i < threadNum; i++)
		{
			threads[i].get();
		}
	}
	else if (m_blenderType == 2)
	{

	}
	else
	{

	}
}

// 如果用户忘记调用该方法，则会在析构函数中进行资源释放
void CCPUBlender::destroyBlender()
{
	if (m_unrollMap != nullptr)
	{
		delete m_unrollMap;
		m_unrollMap = nullptr;
	}
}