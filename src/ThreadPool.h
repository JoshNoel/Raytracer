//
// Created by joshua on 5/16/16.
//

#include <thread>
#include <vector>
#include <glm/vec3.hpp>

#ifndef RAYTRACER_THREADPOOL_H
#define RAYTRACER_THREADPOOL_H


class ThreadPool {
public:
    ThreadPool();
    ~ThreadPool();

	void addJob(const ThreadJob&);

	struct ThreadJob
	{
		int pixelPos[2];
	};
private:
    std::vector<std::thread> threadList;
	short numThreads;



	struct renderThread
	{
	public:
		std::thread m_thread;

		renderThread() {};
		~renderThread()
		{
			if(m_thread.joinable())
				m_thread.join();
		};
	};

};


#endif //RAYTRACER_THREADPOOL_H
