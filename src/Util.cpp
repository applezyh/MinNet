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

ThreadPool* ThreadPool::Interface() {
    static size_t thread_num = std::max(std::thread::hardware_concurrency(), (unsigned int)1);
    static ThreadPool pool(thread_num);
    return &pool;
}

// the constructor just launches some amount of workers
ThreadPool::ThreadPool(size_t threads)
    : stop(false), thread_num(threads)
{
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back(
            [this]
            {
                for (;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            }
            );
}

// the destructor joins all threads
ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers)
        worker.join();
}