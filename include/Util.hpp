#ifndef UTIL_H_
#define UTIL_H_

#include <atomic>
#include <chrono>

class SpinLock {
    std::atomic_flag flag;
public:
    void lock();
    void unlock();
};

class Timer {
    std::chrono::system_clock::time_point time_pt;
    float _cost = 0.f;
public:
    void begin();
    void end();
    float cost();
};

#endif // UTIL_H_
