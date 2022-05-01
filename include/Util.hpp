#ifndef UTIL_H_
#define UTIL_H_

#include <atomic>
#include <chrono>
#include <thread>
#include <future>
#include <functional>
#include <queue>
#include <vector>

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

class ThreadPool {
public:
    static ThreadPool* Interface();
    ThreadPool(size_t);
    template<class F, class... Args>
    std::future<void> enqueue(F&& f, Args&&... args);
    ~ThreadPool();
    size_t thread_num;
private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    // the task queue
    std::queue< std::function<void()> > tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

// add new work item to the pool
template<class F, class... Args>
std::future<void> ThreadPool::enqueue(F&& f, Args&&... args)
{
    auto task = std::make_shared< std::packaged_task<void()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

    std::future<void> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if (stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

#endif // UTIL_H_
