#ifndef BASE_BLENDER_H
#define BASE_BLENDER_H

#include "../utils/UnrollMap.h"
#include "../utils/log.h"

#include <vector>

class CBaseBlender
{
public:
	explicit CBaseBlender();
	virtual ~CBaseBlender();

	virtual void setupBlender() = 0;
	virtual void runBlender(unsigned char* input_data, unsigned char* output_data) = 0;
	virtual void destroyBlender() = 0;
	virtual bool setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned oh, std::string offset, int type) = 0;

protected:
	bool isBase64Decoded(std::string offset);
	std::string trimOffset(std::string offset);
	void splitOffset(std::string& s, char delim, std::vector< std::string >& ret);
	bool isOffsetValid(std::string& _offset);

protected:
	UnrollMap* m_unrollMap;
	// color model: RGB(RGBA)
	unsigned int m_channels;
	float* m_leftMapData;
	float* m_rightMapData;
	unsigned int m_blendWidth;
	unsigned int m_inputWidth;
	unsigned int m_inputHeight;
	unsigned int m_outputWidth;
	unsigned int m_outputHeight;
	std::string m_offset;
	// 1:双鱼眼全景展开map， 2:180度3d展开map， 3：双鱼眼全景展开map，且都位于中间
	int m_blenderType;
	// 表征参数是否发生改变，如果发生改变则需要重新初始化Map实例
	bool m_paramsChanged;
};

#endif

