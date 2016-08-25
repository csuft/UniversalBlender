
#include <iostream>
#include <string>

#include "wrapper/BlenderWrapper.h" 

int main(void)
{
	CBlenderWrapper* wrapper = new CBlenderWrapper;
	int ret = wrapper->capabilityAssessment();

	wrapper->getSingleInstance(); 



	getchar();
	delete wrapper;

	return 0;
}