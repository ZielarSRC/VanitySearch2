#include "Timer.h"
#include <sstream>
#include <iomanip>

std::string Timer::formatTime(double seconds) {
    std::ostringstream oss;
    
    if (seconds < 1.0) {
        oss << std::fixed << std::setprecision(3) << seconds * 1000.0 << " ms";
    } 
    else if (seconds < 60.0) {
        oss << std::fixed << std::setprecision(3) << seconds << " s";
    }
    else {
        int minutes = static_cast<int>(seconds / 60);
        double secs = fmod(seconds, 60.0);
        oss << minutes << " m " << std::fixed << std::setprecision(1) << secs << " s";
    }
    
    return oss.str();
}
