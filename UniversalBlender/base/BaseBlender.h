#ifndef BASE_BLENDER_H
#define BASE_BLENDER_H

#include "../utils/UnrollMap.h"

class CBaseBlender
{
public:
	explicit CBaseBlender();
	virtual ~CBaseBlender();

	virtual void setupBlender() = 0;
	virtual void runBlender() = 0;
	virtual void destroyBlender() = 0;

protected:
	UnrollMap* unrollMap;
	 
};

#endif

