﻿#ifndef _UNROLLMAP_H_
#define _UNROLLMAP_H_

#ifdef _WINDOWS
#ifdef UNROLLMAP_EXPORTS
#define UNROLLMAP_API __declspec(dllexport)
#else
#define UNROLLMAP_API __declspec(dllimport)
#endif
#else
#define UNROLLMAP_API
#endif

#include <string>

class UnrollMapImpl;
class UNROLLMAP_API UnrollMap
{
public:
	UnrollMap();
	~UnrollMap();
	void setOffset(const std::string offset, float fov = 0);
	void init(int inWidth, int inHeight, int outWidth, int outHeight, int type = 1);
	float* getMap(int index);
	float* getCylinderMap(int index);
	float* get3DMap();
	static int getCylinderHeight(int outWidth, float fov = 130.0f);
private:
	UnrollMapImpl* unrollMapImpl;
};

#endif //_UNROLLMAP_H_
