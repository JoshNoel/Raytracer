//
// Created by joshua on 5/16/16.
//

#include "ThreadPool.h"

ThreadPool::ThreadPool()
{
	//TODO Add support for other OSes
	short numCores = std::thread::hardware_concurrency();
	if(numCores == 0)
		numCores = 4;

	this->numThreads = numCores + 1;
}

ThreadPool::~ThreadPool()
{

}