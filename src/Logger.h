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

	static inline std::chrono::duration<double> elapsed()
	{
		return (clock.now() - startTime);// / (long double)(CLOCKS_PER_SEC);
	}

	static inline std::chrono::duration<double> elapsed(std::string label)
	{
		std::chrono::duration<double> e = elapsed();
		timerMap.emplace(label, e);
		return e;
	}

	static void printLog(std::string path, std::string title = "default");
private:
	static std::chrono::steady_clock clock;
	static std::chrono::steady_clock::time_point startTime;
	//Maps desc of time to time
	static std::unordered_map<std::string, std::chrono::duration<double>> timerMap;
};