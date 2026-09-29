// Disk usage via statvfs, with the device name resolved from /proc/self/mountinfo.
#pragma once

#include <string>

#include "types.hpp"

namespace visionai {

class DiskMonitor {
 public:
  // Reports usage for the filesystem containing `path` (default "/").
  DiskSnapshot sample(const std::string& path = "/");

  static DiskSnapshot emptySnapshot(const std::string& path);

  // Extracts the source device for a mount point from /proc/self/mountinfo.
  static std::string resolveDevice(const std::string& path);
};

}  // namespace visionai
