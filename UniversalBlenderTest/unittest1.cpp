#include "stdafx.h"
#include "CppUnitTest.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UniveralBlenderTest
{		
	TEST_CLASS(UnitTest1)
	{
	public:
		
		TEST_METHOD(TestShot)
		{
			// TODO:  �ڴ�������Դ���
			std::cout << "Hello, Zhangzhongke!" << std::endl;
		}

		TEST_METHOD_INITIALIZE(Setup)
		{
			std::cout << "Hello, C++ world!" << std::endl;
		}

		TEST_METHOD_CLEANUP(TearDown)
		{
			std::cout << "Hello, World!" << std::endl;
		}
	};
}