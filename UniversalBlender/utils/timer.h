#ifndef _TIMER_H_
#define _TIMER_H_

/**
 * \file Timer.h
 * \brief A timer class that provides a cross platform timer for use
 * in timing code progress with a high degree of accuracy.
 */
#if (defined _WIN32 || defined _WIN64)
/**
 * \typedef __int64 i64
 * \brief Maps the windows 64 bit integer to a uniform name
 */
#if defined(__MINGW64__) || defined(__MINGW32__)
typedef long long i64;
#else
typedef __int64 i64;
#endif
#else
/**
 * \typedef long long i64
 * \brief Maps the linux 64 bit integer to a uniform name
 */
typedef long long i64;
#endif


class Timer {

public:
    Timer();
    ~Timer();
    void Start(void);
    void Stop(void);
    void Reset(void);
    double GetElapsedTime(void);
	unsigned long PthreadSelf();
    void current_utc_time(struct timespec* ts);
private:

    i64 _freq;
    i64 _clocks;
    i64 _start;
};



#endif // _TIMER_H_

