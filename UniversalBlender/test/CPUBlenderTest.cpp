#include "CPUBlenderTest.h"

CCPUBlenderTest::CCPUBlenderTest()
{
}


CCPUBlenderTest::~CCPUBlenderTest()
{
}

CppUnit::Test* CCPUBlenderTest::suite()
{
	CppUnit::TestSuite* suiteOfTest = new CppUnit::TestSuite("Test offset validator");
	suiteOfTest->addTest(new CppUnit::TestCaller<CCPUBlenderTest>("Test 1 - test isBase64Decoded", &CCPUBlenderTest::test_isBase64Decoded));
	suiteOfTest->addTest(new CppUnit::TestCaller<CCPUBlenderTest>("Test 2 - test isOffsetValid", &CCPUBlenderTest::test_isOffsetValid));

	return nullptr;
}

void CCPUBlenderTest::test_isBase64Decoded()
{
	std::string offset;
	bool ret;
	offset = "2_674.55_2160.15_724.35_0.00_0.00_180_676.50_717.25_714.00_0.00_0.00_0.36_2880_1440";
	ret = isBase64Decoded(offset);
	CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE("Offset is decoded", CPPUNIT_ASSERT(ret == true));

	offset = "Ml83MTMuNjQ5XzcxMy45MzNfNzA4LjA1Xy0yLjY1Xy0xLjU5OV8tMC4zNThfNzEyLjQwM18yMTY1LjVfNzI3LjU4M18xNzcuMzVfLTE4MC41NF8wLjE3Nl8yODgwXzE0NDBfNTE2";
	ret = isBase64Decoded(offset);
	CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE("Offset is decoded", CPPUNIT_ASSERT(ret == false));
}

void CCPUBlenderTest::test_isOffsetValid()
{
	std::string offset;
	bool ret;

	// invalid format offset
	offset = " zhangzhongke_is_a_good_boy ";
	ret = isOffsetValid(offset);
	CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE("Offset is invalid", CPPUNIT_ASSERT(ret == false));

	// valid format offset
	offset = "2_674.55_2160.15_724.35_0.00_0.00_180_676.50_717.25_714.00_0.00_0.00_0.36_2880_1440";
	ret = isOffsetValid(offset);
	CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE("Offset is valid", CPPUNIT_ASSERT(ret == true));

	// prefix spaces and postfix spaces
	offset = " 2_674.55_2160.15_724.35_0.00_0.00_180_676.50_717.25_714.00_0.00_0.00_0.36_2880_1440 ";
	ret = isOffsetValid(offset);
	CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE("Offset is valid", CPPUNIT_ASSERT(ret == true));

	// valid format offset
	offset = "2_674.55_2160.15_724.35_0.00_0.00_180_676.50_717.25_714.00_0.00_0.00_0.36_2880_1440_52";
	ret = isOffsetValid(offset);
	CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE("Offset is valid", CPPUNIT_ASSERT(ret == true));

	// invalid format offset
	offset = "2_674.55_2160.15_724.35_0.00_0.00_180_676.50_717.25_714.00_0.00_0.00";
	ret = isOffsetValid(offset);
	CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE("Offset is invalid", CPPUNIT_ASSERT(ret == false));

	// Base64 encoded offset
	offset = "Ml83MTMuNjQ5XzcxMy45MzNfNzA4LjA1Xy0yLjY1Xy0xLjU5OV8tMC4zNThfNzEyLjQwM18yMTY1LjVfNzI3LjU4M18xNzcuMzVfLTE4MC41NF8wLjE3Nl8yODgwXzE0NDBfNTE2";
	ret = isOffsetValid(offset);
	CPPUNIT_ASSERT_ASSERTION_FAIL_MESSAGE("Offset is invalid", CPPUNIT_ASSERT(ret == false));
}