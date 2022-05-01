#include "Util.hpp"

void SpinLock::lock() {
    while (flag.test_and_set(std::memory_order_acquire));
}

void SpinLock::unlock() {
    flag.clear(std::memory_order_release);
}

void Timer::begin() {
    time_pt = std::chrono::system_clock::now();
}
void Timer::end() {
    auto end = std::chrono::system_clock::now();
    auto duration = duration_cast<std::chrono::microseconds>(end - time_pt);
    _cost = float(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}
float Timer:: cost() {
    float ret = _cost;
    _cost = 0.f;
    return ret;
}