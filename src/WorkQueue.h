#ifndef RAYTRACER_WORKQUEUE_H
#define RAYTRACER_WORKQUEUE_H

#include <list>
#include <mutex>
#include <condition_variable>

template<typename T>
class WorkQueue
{
public:
	WorkQueue(int max_size) 
		: mutex(), condition_variable(), list(), doneWork(false), maxSize(max_size)
	{}
	WorkQueue(WorkQueue&& wq)
	{
		wq.mutex.lock();
		doneWork = std::move(wq.doneWork);
		list = std::move(wq.list);
		wq.mutex.unlock();
	}
	~WorkQueue() 
	{
		mutex.lock();
		doneWork = true;
		mutex.unlock();
	}

	void push(const T& val)
	{
		//lock for push onto queue
		list.push_back(val);
		condition_variable.notify_one();
	}

	T pop()
	{
		//lock the mutex
		//pop front
		T t = list.front();
		list.pop_front();
		return t;
	}

	void clear()
	{
		list.clear();
	}

	size_t size()
	{
		return list.size();
	}

	size_t max_size()
	{
		return maxSize;
	}

	std::mutex mutex;
	std::condition_variable condition_variable;
	bool doneWork;

private:
	std::list<T> list;
	size_t maxSize;
};

#endif
