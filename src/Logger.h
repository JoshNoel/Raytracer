#pragma once
#include <chrono>
#include <string>
#include <unordered_map>

class Logger
{
public:
	Logger();
	~Logger();

	static inline void startClock()
	{
		startTime = clock.now();
	}

	//return elapsed time in steady_clock ticks
	static inline void record(std::string label)
	{
		timerMap.emplace(label, elapsed());
	}

	//appends currently recorded log to the .txt file at path
		//logger ends at current time
		//title used to identify log within the logger .txt file
	static void printLog(std::string path, std::string title = "default");
private:

	//return elapsed stead_clock ticks
	static inline std::chrono::duration<float, std::chrono::seconds::period> elapsed()
	{
		return (clock.now() - startTime);
	}

	static std::chrono::steady_clock clock;
	static std::chrono::steady_clock::time_point startTime;

	//Map: description of time TO the time itself
	static std::unordered_map<std::string, std::chrono::duration<float, std::chrono::seconds::period>> timerMap;
};