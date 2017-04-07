#include "BaseBlender.h"

#include "../utils/base64.h"
#include <regex>

CBaseBlender::CBaseBlender() : m_channels(4), m_blendWidth(0), m_leftMapData(nullptr), m_rightMapData(nullptr), m_unrollMap(nullptr),
m_inputHeight(0), m_inputWidth(0), m_outputWidth(0), m_outputHeight(0), m_paramsChanged(false), m_blenderType(1), m_colorMode(0)
{
}


CBaseBlender::~CBaseBlender()
{
	// leave it alone
}

void CBaseBlender::setupBlender()
{
	if (m_paramsChanged)
	{
		if (m_unrollMap != nullptr)
		{
			delete m_unrollMap;
			m_unrollMap = nullptr;
			m_leftMapData = nullptr;
			m_rightMapData = nullptr;
		}
		m_unrollMap = new UnrollMap;
		if (m_blenderType == 1)    // panoramic blender
		{
			m_unrollMap->setOffset(m_offset); 
			m_unrollMap->init(m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight);
			m_leftMapData = m_unrollMap->getMap(0);
			m_rightMapData = m_unrollMap->getMap(1);
		}
		else if(m_blenderType == 2)// 3d blender
		{
			m_unrollMap->setOffset(m_offset, 200.0f);
			m_unrollMap->init(m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight, 3);
			m_leftMapData = m_unrollMap->getMap(0);
			m_rightMapData = m_unrollMap->getMap(1);
		}
		else if (m_blenderType == 4)
		{
			m_unrollMap->setOffset(m_offset);
			m_unrollMap->init(m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight, 4);
			m_leftMapData = m_unrollMap->get3DMap();
		}
		else                       // panoramic cylinder blender
		{
			m_unrollMap->setOffset(m_offset);
			m_unrollMap->init(m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight);
			m_leftMapData = m_unrollMap->getCylinderMap(0);
			m_rightMapData = m_unrollMap->getCylinderMap(1);
		}

		m_paramsChanged = false;
	}
}

bool CBaseBlender::setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned oh, std::string offset, int type)
{
	if (iw <= 0 || ih <= 0 || ow <= 0 || oh <= 0)
	{
		LOGERR("Invalid resolution parameters, please check again carefully!");
		return false;
	}

	if (!isOffsetValid(offset))
	{
		LOGERR("Invalid offset format, please check again carefully!");
		return false;
	}

	if (iw != m_inputWidth || ih != m_inputHeight || ow != m_outputWidth || oh != m_outputHeight || m_offset.compare(offset))
	{
		m_inputWidth = iw;
		m_inputHeight = ih;
		m_outputWidth = ow;
		m_outputHeight = oh;
		m_offset = offset;
		// To indicate the parameters have changed.
		m_paramsChanged = true;
	}
	m_blenderType = type;

	return true;
}

int CBaseBlender::getCylinderOutputHeight(int output_width, float fov)
{ 
	return UnrollMap::getCylinderHeight(output_width, fov);
}

/**
* Offset判断offset是否已经经过Base64解码
* Base64编码方案最好的讲解：http://www.cnblogs.com/chengxiaohui/articles/3951129.html
*/
bool CBaseBlender::isBase64Decoded(std::string offset)
{
	if (offset.empty())
	{
		return false;
	}
	std::size_t found = offset.find('_');
	return found == std::string::npos ? false : true;
}

/**
* 去掉offset前后非法的部分
*/
std::string CBaseBlender::trimOffset(std::string offset)
{
	if (offset.empty())
	{
		return std::string("");
	}

	const char* START_TEMPLATE = "0123456789-";
	const char* END_TEMPLATE = "0123456789";

	// 剔除头部的非法字符
	std::size_t start;
	start = offset.find_first_of(START_TEMPLATE);

	// 剔除尾部的非法字符
	std::size_t end;
	end = offset.find_last_of(END_TEMPLATE);

	return std::string(offset, start, end - start + 1);
}

/**
* 分割offset，并对offset的每个部分进行判断
*/
void CBaseBlender::splitOffset(std::string& s, char delim, std::vector< std::string >& ret)
{
	size_t last = 0;
	size_t index = s.find_first_of(delim, last);
	while (index != std::string::npos)
	{
		ret.push_back(s.substr(last, index - last));
		last = index + 1;
		index = s.find_first_of(delim, last);
	}
	if (index - last > 0)
	{
		ret.push_back(s.substr(last, index - last));
	}
}

/**
* 通过判断offset中参数的个数来确定是否合法
*/
bool CBaseBlender::isOffsetValid(std::string& _offset)
{
	LOGINFO("Offset: %s", _offset.c_str());
	// to be fix
	//////////////////////////////////////////////////////////////////////////
	std::string temp;
	char* decoded_offset = nullptr;
	// 如还未进行Base64解码，则先解码
	if (!isBase64Decoded(_offset))
	{
		decoded_offset = base64Decode(_offset.c_str());
		temp = decoded_offset;

		// The caller must free the memory allocated by the decoder.
		delete[] decoded_offset;
	}
	else
	{
		temp = _offset;
	}
	//////////////////////////////////////////////////////////////////////////
	_offset = trimOffset(temp);
	//LOGINFO("Trimmed offset: %s", _offset.c_str());
	std::vector<std::string> result;
	splitOffset(_offset, '_', result);
	std::vector<std::string>::iterator itb = result.begin();
	std::string::size_type sz;
	// 镜头数目：目前主要是两个镜头的相机，将来可能有其他变化
	int camerasNum = std::stoi(*itb, &sz);
	if (camerasNum <= 1 || camerasNum > 6)
	{
		return false;
	}
	// offset中应包含参数个数
	int paramsNum = camerasNum * 6 + 2;
	bool ret;
	while (itb != result.end())
	{
		ret = std::regex_match(*itb, std::regex("-?[0-9]{1,}\\.?[0-9]{0,}"));
		if (!ret)
		{
			return false;
		}
		++itb;
	}

	return result.size() >= paramsNum ? true : false;
}

void CBaseBlender::RGB2RGBA(unsigned char* rgba, unsigned char* rgb, int imageSize)
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

void CBaseBlender::RGBA2RGB(unsigned char* rgb, unsigned char* rgba, int imageSize)
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

void CBaseBlender::startTimer()
{
#ifdef DEBUG
	m_timer.Reset();
	m_timer.Start();
#endif
}

void CBaseBlender::stopTimer(const char* operation)
{
#ifdef DEBUG
	m_timer.Stop();
	LOGINFO("%s consumes %lf(ms)", operation, m_timer.GetElapsedTime()*1000);
#endif
}