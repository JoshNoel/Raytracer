//
// Created by joshua on 5/16/16.
//

#include "Renderer.h"
#include "ThreadPool.h"
#include "WorkQueue.h"
#include <iostream>

///Creates threads based on number of cores available
ThreadPool::ThreadPool(const Renderer* renderer)
	: jobQueue(800*800)
{
	//TODO Add support for other OSes
	short numCores = std::thread::hardware_concurrency();
	if(numCores == 0)
		numCores = 4;	

	this->numThreads = numCores;

	//construct all threads then start to avoid threads starting before renderThread is fully constructed
	for(int i = 0; i < numThreads; i++)
	{
		threadList.emplace_back(&jobQueue, renderer);
	}
#ifndef USE_CUDA
	for (int i = 0; i < numThreads; i++) 
	{
		threadList[i].start();
	}
#endif
}

ThreadPool::~ThreadPool()
{
	std::unique_lock<std::mutex> lock(jobQueue.mutex);
	jobQueue.doneWork = true;
	jobQueue.clear();
	jobQueue.condition_variable.notify_all();
	lock.unlock();
}

///Adds a job to the workQueue and notifies a waiting thread
bool ThreadPool::addJob(const ThreadJob& job)
{
	//lock queue mutex
	//notify waiting thread job is available
	std::lock_guard<std::mutex> lock(jobQueue.mutex);
	if(jobQueue.size() < jobQueue.max_size())
	{
		jobQueue.push(job);
		jobQueue.condition_variable.notify_one();
		return true;
	}
	else
	{
		return false;
	}
}

void ThreadPool::joinThreads()
{
	for (unsigned i = 0; i < threadList.size(); i++)
	{
		threadList[i].m_thread.join();
	}
}

void ThreadPool::doneAddingJobs()
{
	jobQueue.doneWork = true;
}

void ThreadPool::RenderThread::run()
{
	while(true)
	{
		////check queue for jobs////
			//wait until a job is added (jobQueue size > 0) or work is done (doneWork == true)
		try
		{
			std::unique_lock<std::mutex> lock(m_jobQueue->mutex);
			m_jobQueue->condition_variable.wait(lock, [this] {return (this->m_jobQueue->size() != 0 || this->m_jobQueue->doneWork);; });
		

			//if a job is added and the thread is notified render the pixel
			bool done = m_jobQueue->doneWork && (this->m_jobQueue->size() == 0);
			if(!done)
			{
				ThreadJob job = m_jobQueue->pop();
				lock.unlock();
				//render pixel
#ifndef USE_CUDA
				glm::vec3 color = p_renderer->renderPixel(job.x(), job.y());
				p_renderer->writeImage(color, job.x(), job.y());
#endif USE_CUDA
			}
			else
			{
				break;
			}
		}
		catch (std::exception e)
		{
			std::cout << "Error while renderThread was waiting: " << e.what();
		}
	}
}
