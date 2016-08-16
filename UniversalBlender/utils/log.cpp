#define _CRT_SECURE_NO_WARNINGS

#include <cstring>
#include <time.h>
#include <sys/stat.h>
#include "log.h" 
#define _VERSION_ "1.0.0"

CMyLog::CMyLog()
	: m_fp(NULL)
	, m_ulFileSize(0)
{
	m_ulFileSize = 5 * 1024 * 1024; /* 5MB per log file */
 
#if (defined _WIN32 || defined _WIN64)
	m_hMutex = CreateMutex(NULL, false, NULL);
	if (!m_hMutex)
	{
		printf("createmutex fail.\n");
		return;
	}
#else
	pthread_mutex_init(&m_mutex, NULL);
#endif

#if (defined _WIN32 || defined _WIN64)
	char* home = getenv("HOMEDRIVE");
	if (!home)
	{
		printf("cann't get home root\n");
		return;
	}

	m_path = std::string(home);

	home = getenv("HOMEPATH");
	if (!home)
	{
		printf("cann't get home\n");
	}
	m_path = m_path + home;

	m_path = m_path + "\\AppData\\Local";
	if (_access(m_path.c_str(), 0))
	{
		printf("HOME\\AppData\\Local dir not exist\n");
		return;
	}

	m_path = m_path + "\\insta360";
	if (_access(m_path.c_str(), 0))
	{
		_mkdir(m_path.c_str());
	}

	m_path = m_path + "\\UniversalBlender";
	if (_access(m_path.c_str(), 0))
	{
		_mkdir(m_path.c_str());
	}

	std::string logfile = m_path + "\\Blender.log";
	m_fp = fopen(logfile.c_str(), "a+");
#else
	char* home = getenv("HOME");
	if (!home)
	{
		printf("cann't get home dir\n");
		return;
	}

	m_path = std::string(home);
	m_path = m_path + "/Library/Application Support";
	if (access(m_path.c_str(), 0))
	{
		printf("HOME/Library/Application Support dir not exist\n");
		return;
	}

	m_path = m_path + "/insta360";
	if (access(m_path.c_str(), 0))
	{
		mkdir(m_path.c_str(), 0766);
	}

	m_path = m_path + "/UniversalBlender";
	if (access(m_path.c_str(), 0))
	{
		mkdir(m_path.c_str(), 0766);
	}

	std::string logfile = m_path + "/Blender.log";
	m_fp = fopen(logfile.c_str(), "a+");

#endif
	
	if (!m_fp)
	{
		printf("open log file fail\n");
		return;
	}
}

CMyLog::~CMyLog()
{
	if (m_fp)
	{
		fclose(m_fp);
	}

#if (defined _WIN32 || defined _WIN64)
	CloseHandle(m_hMutex);
#else
	pthread_mutex_destroy(&m_mutex);
#endif

	printf("log destroy\n");
}

CMyLog& CMyLog::GetInstance()
{
	static CMyLog* pLog = NULL;

	if (!pLog)
	{
		pLog = new CMyLog;
		pLog->Log(LOG_LEVEL_INFO, __FILE__, __LINE__, "Verision: %s, build date: %s, time:%s", _VERSION_, __DATE__, __TIME__);
	}

	return *pLog;
}

void CMyLog::Log(unsigned char level, const char* file, int line, const char* fmt, ...)
{
	static char LogLevelString[][10] = {"ERROR", "INFO ", "DEBUG"};

	if (level > m_ucLevel)
	{
		return;
	}

#if (defined _WIN32 || defined _WIN64)
	WaitForSingleObject(m_hMutex, INFINITE);
#else
	pthread_mutex_lock(&m_mutex);
#endif

	/* to stdout */
	time_t timer = time(NULL);
	struct tm *tmt = localtime(&timer);
	va_list arg;

	/* to log file */
	if (m_fp)
	{
		fprintf(m_fp, "%04d-%02d-%02d %02d:%02d:%02d %s ",
			tmt->tm_year + 1900,
			tmt->tm_mon + 1,
			tmt->tm_mday,
			tmt->tm_hour,
			tmt->tm_min,
			tmt->tm_sec,
			LogLevelString[level]);

		/* log content */
		va_start(arg, fmt);
		vfprintf(m_fp, fmt, arg);
		va_end(arg);

		/* log file */
		fprintf(m_fp, "%s:%d\n", file, line);

		fflush(m_fp);

		ChangeLogFile();
	}

#if (defined _WIN32 || defined _WIN64)
	ReleaseMutex(m_hMutex);
#else
	pthread_mutex_unlock(&m_mutex);
#endif
}

void CMyLog::ChangeLogFile()
{
	std::string origfile = m_path + "/Blender.log";

	struct stat statbuff;
	if (stat(origfile.c_str(), &statbuff) < 0)
	{
		return;
	}

	if (statbuff.st_size  < (int)m_ulFileSize)
	{
		return;
	}

	fclose(m_fp);
	m_fp = NULL;

	char filename[256] = {0};
	time_t timer = time(NULL);
	struct tm *tmt = localtime(&timer);

	SPRINTF(filename, 256 - 1, "/Blender.log.%04d_%02d_%02d_%02d_%02d_%02d",
		tmt->tm_year + 1900,
		tmt->tm_mon + 1,
		tmt->tm_mday,
		tmt->tm_hour,
		tmt->tm_min,
		tmt->tm_sec);

	std::string newfile = m_path + filename;
	rename(origfile.c_str(), newfile.c_str());
	m_fp = fopen(origfile.c_str(), "a+");
	if (!m_fp)
	{
		printf("open log file fail\n");
	}
}

