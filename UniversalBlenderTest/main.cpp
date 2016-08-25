
#include <iostream>
#include <string>

#include "wrapper/BlenderWrapper.h" 

int main(void)
{
	CBlenderWrapper* wrapper = new CBlenderWrapper;
	int ret = wrapper->capabilityAssessment();
	wrapper->getSingleInstance(); 

	// 1. capabilityAssessment()
	// 2. getSingleInstance();
	// 3. initializeDevice();
	// 4. runImageBlender();


	getchar();
	delete wrapper;

	return 0;
}