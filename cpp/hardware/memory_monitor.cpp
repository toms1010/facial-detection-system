#include "memory_monitor.hpp"

#include <algorithm>

#include "sysfs.hpp"

namespace visionai {

std::uint64_t MemoryMonitor::parseMemInfo(const std::string& content, const std::string& key) {
  const std::string needle = key + ":";
  for (const auto& line : sysfs::split(content, "\n")) {
    if (line.rfind(needle, 0) != 0) {
      continue;
    }
    const auto fields = sysfs::split(line, " \t");
    if (fields.size() < 2) {
      continue;
    }
    const auto value = sysfs::parseUint(fields[1]);
    if (!value) {
      continue;
    }
    const bool isKb = fields.size() > 2 && fields[2] == "kB";
    return isKb ? *value * 1024ULL : *value;
  }
  return 0;
}

double MemoryMonitor::usagePercent(const MemorySnapshot& snapshot) {
  if (snapshot.totalBytes == 0) {
    return 0.0;
  }
  const double used = static_cast<double>(snapshot.totalBytes - snapshot.availableBytes);
  return std::clamp(used / static_cast<double>(snapshot.totalBytes) * 100.0, 0.0, 100.0);
}

MemorySnapshot MemoryMonitor::sample() {
  MemorySnapshot snapshot;
  const auto content = sysfs::readFile("/proc/meminfo");
  if (!content) {
    return snapshot;
  }

  snapshot.totalBytes = parseMemInfo(*content, "MemTotal");
  snapshot.freeBytes = parseMemInfo(*content, "MemFree");
  snapshot.availableBytes = parseMemInfo(*content, "MemAvailable");
  snapshot.cachedBytes = parseMemInfo(*content, "Cached");
  snapshot.buffersBytes = parseMemInfo(*content, "Buffers");
  snapshot.swapTotalBytes = parseMemInfo(*content, "SwapTotal");
  snapshot.swapFreeBytes = parseMemInfo(*content, "SwapFree");

  if (snapshot.availableBytes == 0) {
    snapshot.availableBytes =
        snapshot.freeBytes + snapshot.cachedBytes + snapshot.buffersBytes;
  }
  if (snapshot.totalBytes >= snapshot.availableBytes) {
    snapshot.usedBytes = snapshot.totalBytes - snapshot.availableBytes;
  } else {
    snapshot.usedBytes = 0;
  }

  snapshot.usagePercent = usagePercent(snapshot);
  if (snapshot.swapTotalBytes > 0) {
    const double swapUsed = static_cast<double>(snapshot.swapTotalBytes - snapshot.swapFreeBytes);
    snapshot.swapUsagePercent = std::clamp(
        swapUsed / static_cast<double>(snapshot.swapTotalBytes) * 100.0, 0.0, 100.0);
  }
  return snapshot;
}

}  // namespace visionai
