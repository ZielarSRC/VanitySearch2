#pragma once
#include <chrono>
#include <string>

class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    
    Timer() : startTime(Clock::now()) {}
    
    void reset() { 
        startTime = Clock::now(); 
    }
    
    double elapsed() const {
        return std::chrono::duration<double>(Clock::now() - startTime).count();
    }
    
    static std::string formatTime(double seconds);
    
private:
    TimePoint startTime;
};
