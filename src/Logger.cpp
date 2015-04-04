#include "Logger.h"
#include <fstream>

Logger::Logger()
{
}

Logger::~Logger()
{

}

void Logger::printLog(std::string path, std::string title)
{
	std::ofstream os;
	os.open(path.c_str(), std::ios_base::app);
	os << "===========================" << std::endl;
#ifdef NDEBUG
	os << title << ": " << "RELEASE" << std::endl << std::endl;
#else
	os << title << ": " << "DEBUG" << std::endl << std::endl;
#endif
	if(timerMap.size() != 0)
	{
		std::hash_map<std::string, std::chrono::duration<double>>::iterator i = timerMap.begin();
		while(i != timerMap.end())
		{
			os << i->first << ": " << std::chrono::duration_cast<std::chrono::microseconds>(i->second).count() << std::endl;
			i++;
		}
	}
}

std::chrono::steady_clock Logger::clock;
std::chrono::steady_clock::time_point Logger::startTime;
//Maps desc of time to time
std::hash_map<std::string, std::chrono::duration<double>> Logger::timerMap;
