#include "Logger.h"
#include <fstream>

void Logger::printLog(std::string path)
{
	if (enabled)
	{
		if (title == "")
			title = Logger::DEFAULT_TITLE;
		std::ofstream os;
		os.open(path.c_str(), std::ios_base::app);
		os << "===========================" << std::endl;
#ifndef _DEBUG
		os << title << ": " << "RELEASE" << std::endl << std::endl;
#else
		os << title << ": " << "DEBUG" << std::endl << std::endl;
#endif
		if (timerMap.size() != 0)
		{
			std::unordered_map<std::string, std::chrono::duration<float, std::chrono::seconds::period>>::iterator i = timerMap.begin();
			while (i != timerMap.end())
			{
				os << i->first << ": " << i->second.count() << std::endl;
				++i;
			}
		}
		os << std::endl;
	}
}

//Maps desc of time to time
bool Logger::enabled = false;
std::string Logger::title;
const std::string Logger::DEFAULT_TITLE = "Default Title";
std::unordered_map<std::string, std::chrono::duration<float, std::chrono::seconds::period>> Logger::timerMap;
std::unordered_map<std::string, std::chrono::time_point<std::chrono::steady_clock>> Logger::startTimeMap;

