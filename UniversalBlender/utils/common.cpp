#include "common.h"

void flipColorMode(void* buffer)
{
	if (buffer == nullptr)
	{
		return;
	}

}

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