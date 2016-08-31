#ifndef _COMMON_H
#define _COMMON_H

// Convert RGB to BGR or reverse
void flipColorMode(void* buffer);
// Convert RGB to RGBA
void RGB2RGBA(unsigned char* rgba, unsigned char* rgb, int imageSize);
// Convert RGBA to RGB
void RGBA2RGB(unsigned char* rgb, unsigned char* rgba, int imageSize);

#endif


