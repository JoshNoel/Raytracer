#include "Core.h"
#include "CudaLoader.h"

///Creates ThreadPool from renderer
Core::Core(Renderer* renderer)
	: threadPool(renderer), renderer(renderer)
{
	
}

Core::~Core()
{}

void Core::render()
{
#ifdef USE_CUDA
	//if using cuda launch kernel
	if (renderer->init())
		renderer->renderCuda();
	else
		std::cerr << "Error initializing renderer!" << std::endl;

#else
	std::cout << "starting rendering..." << std::endl;
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
#endif

}


