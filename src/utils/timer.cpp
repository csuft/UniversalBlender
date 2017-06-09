
#include "timer.h"

#if (defined _WIN32 || defined _WIN64)
#include <windows.h>
#else
#include <mach/clock.h>
#include <mach/mach.h>
#include <sys/time.h>
#include <time.h>
#include <pthread.h> 
#endif

Timer::Timer() : _clocks(0), _start(0)
{

#if (defined _WIN32 || defined _WIN64)
    QueryPerformanceFrequency((LARGE_INTEGER *)&_freq);
#else
    _freq = 1000;
#endif

}

Timer::~Timer()
{
    // EMPTY!
}

void
Timer::Start(void)
{

#if (defined _WIN32 || defined _WIN64)
    QueryPerformanceCounter((LARGE_INTEGER *)&_start);
#else
    struct timespec s;
    current_utc_time(&s);
    _start = (i64)s.tv_sec * 1e9 + (i64)s.tv_nsec;
#endif

}

void
Timer::Stop(void)
{
    i64 n;

#if (defined _WIN32 || defined _WIN64)
    QueryPerformanceCounter((LARGE_INTEGER *)&n);
#else
    struct timespec s;
    current_utc_time(&s);
    n = (i64)s.tv_sec * 1e9 + (i64)s.tv_nsec;
#endif

    n -= _start;
    _start = 0;
    _clocks += n;
}

void
Timer::Reset(void)
{

    _clocks = 0;
}

double
Timer::GetElapsedTime(void)
{
#if (defined _WIN32 || defined _WIN64)
    return (double)_clocks / (double) _freq;
#else
    return (double)_clocks / (double) 1e9;
#endif

}


unsigned long Timer::PthreadSelf()
{
#if (defined _WIN32 || defined _WIN64)
	return  ::GetCurrentThreadId();
#else 
	return  (unsigned long)pthread_self();
#endif 
}

void Timer::current_utc_time(struct timespec* ts)
{
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts->tv_sec = mts.tv_sec;
  ts->tv_nsec = mts.tv_nsec;
#endif
}
