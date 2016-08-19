#ifndef BASE_BLENDER_H
#define BASE_BLENDER_H

#include "../utils/UnrollMap.h"
#include "../utils/log.h"

class CBaseBlender
{
public:
	explicit CBaseBlender();
	virtual ~CBaseBlender();

	virtual void setupBlender() = 0;
	virtual void runBlender(const unsigned char* input_data, unsigned char* output_data, int type) = 0;
	virtual void destroyBlender() = 0;
	virtual void setParams(const unsigned int iw, const unsigned int ih, const unsigned int ow, const unsigned oh, std::string offset) = 0;

protected:
	bool isBase64Decoded(std::string offset);
	std::string trimOffset(std::string offset);
	void splitOffset(std::string& s, char delim, std::vector< std::string >& ret);
	bool isOffsetValid(std::string& _offset);

protected:
	UnrollMap* unrollMap;
};

#endif

