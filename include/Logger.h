#pragma once
#include <chrono>
#include <string>
#include <unordered_map>

class Logger
{
public:
	Logger() = delete;
	~Logger() = delete;

	static inline void startClock(std::string label)
	{
		startTimeMap.emplace(label, std::chrono::steady_clock::now());
	}

	//return elapsed time in steady_clock ticks
	static inline void record(std::string label)
	{
		timerMap.emplace(label, elapsed(label));
	}

	//appends currently recorded log to the .txt file at path
		//logger ends at current time
		//title used to identify log within the logger .txt file
	static void printLog(std::string path);
	static bool enabled;
	static std::string title;
	static const std::string DEFAULT_TITLE;
private:

	//return elapsed stead_clock ticks
	static inline std::chrono::duration<float, std::chrono::seconds::period> elapsed(std::string label)
	{
		return (std::chrono::steady_clock::now() - startTimeMap[label]);
	}

	//Map: description of time TO the time itself
	static std::unordered_map<std::string, std::chrono::time_point<std::chrono::steady_clock>> startTimeMap;
	static std::unordered_map<std::string, std::chrono::duration<float, std::chrono::seconds::period>> timerMap;
};