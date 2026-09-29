// Memory statistics from /proc/meminfo.
#pragma once

#include <string>

#include "types.hpp"

namespace visionai {

class MemoryMonitor {
 public:
  MemorySnapshot sample();

  // Parses the "key:  value kB" lines of /proc/meminfo into bytes.
  static std::uint64_t parseMemInfo(const std::string& content, const std::string& key);

  static double usagePercent(const MemorySnapshot& snapshot);
};

}  // namespace visionai
