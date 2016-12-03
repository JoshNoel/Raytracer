#include "Core.h"

///Creates ThreadPool from renderer
Core::Core(const Renderer* renderer)
	: threadPool(renderer), renderer(renderer)
{
	
}

Core::~Core()
{}

void Core::render()
{
	for(int i = 0; i < renderer->image->width; i++)
	{
		for(int j = 0; j < renderer->image->height; j++)
		{
			bool added = false;
			while(!added)
			{
				if(threadPool.addJob(ThreadPool::ThreadJob(i, j)))
					added = true;			
			}
		}
	}
	threadPool.doneAddingJobs();
	threadPool.joinThreads();
}


