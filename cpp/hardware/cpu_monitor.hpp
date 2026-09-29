// CPU statistics from /proc/stat, /proc/cpuinfo and /proc/loadavg.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "types.hpp"

namespace visionai {

class CpuMonitor {
 public:
  CpuMonitor();

  // Samples /proc/stat and computes deltas against the previous sample.
  // The first call only establishes a baseline and reports 0% usage.
  CpuSnapshot sample();

  const CpuSnapshot& last() const { return last_; }

  static CpuTimes parseStatLine(const std::string& line);
  static std::string parseModelName(const std::string& cpuinfo);
  static std::vector<double> parseLoadAverage(const std::string& text);

 private:
  CpuSnapshot last_;
  CpuTimes previousTotal_;
  std::vector<CpuTimes> previousCores_;
  bool primed_ = false;
};

}  // namespace visionai
