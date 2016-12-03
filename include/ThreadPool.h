//
// Created by joshua on 5/16/16.
//

#include <thread>
#include <vector>
#include "WorkQueue.h"
#include "Renderer.h"
#include <iostream>

#ifndef RAYTRACER_THREADPOOL_H
#define RAYTRACER_THREADPOOL_H


class ThreadPool {
public:
	struct ThreadJob
	{
	public:
		enum JOB_TYPE
		{RENDER, WRITE};
		ThreadJob(int x, int y, JOB_TYPE type = RENDER)
			: type(type)
		{
			data[0] = x;
			data[1] = y;
		}
		//ThreadJob(float r, float g, float b, JOB_TYPE = WRITE);
		//TODO generalize threadjob to allow for rendering and writing jobs
		~ThreadJob()
		{
		}

		int x() { return data[0]; }
		int y() { return data[1]; }
		JOB_TYPE getType() { return type; }

	private:
		//pixelData: x -> 0, y -> 1
		//color data: r -> 0, g -> 1, b -> 2
		int data[2];
		JOB_TYPE type;
	};

    ThreadPool(const Renderer*);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& ThreadPool::operator=(const ThreadPool&) = delete;
    ThreadPool& ThreadPool::operator=(const ThreadPool&&) = delete;

	bool addJob(const ThreadJob&);
	void joinThreads();
	void doneAddingJobs();
	
private:
	struct RenderThread
	{
	public:
		std::thread m_thread;

		RenderThread(WorkQueue<ThreadJob>* wq, const Renderer* r)
			: m_jobQueue(wq), p_renderer(r), m_thread()
		{
		}

		RenderThread(RenderThread&& rt)
			: m_jobQueue(rt.m_jobQueue), 
			p_renderer(rt.p_renderer),
			m_thread(std::move(rt.m_thread))
		{
			rt.m_jobQueue = 0;
			rt.p_renderer = 0;
		}

		~RenderThread()
		{
			if(m_thread.joinable())
				m_thread.join();
		};

		void start() 
		{ 
			try
			{
				m_thread = std::thread(&RenderThread::run, this);
			}
			catch(std::exception e)
			{
				std::cout << e.what() << std::endl;
			}
		}
	private:
		void run();
		WorkQueue<ThreadJob>* m_jobQueue;
		const Renderer* p_renderer;
	};

	//map thread with output list
    std::vector<RenderThread> threadList;

	WorkQueue<ThreadJob> jobQueue;
	short numThreads;

};


#endif //RAYTRACER_THREADPOOL_H
