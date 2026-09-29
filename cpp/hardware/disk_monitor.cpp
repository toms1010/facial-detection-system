#include "disk_monitor.hpp"

#include <sys/statvfs.h>

#include <algorithm>
#include <filesystem>

#include "sysfs.hpp"

namespace fs = std::filesystem;

namespace visionai {

DiskSnapshot DiskMonitor::emptySnapshot(const std::string& path) {
  DiskSnapshot snapshot;
  snapshot.mountPoint = path;
  return snapshot;
}

std::string DiskMonitor::resolveDevice(const std::string& path) {
  const auto mountinfo = sysfs::readFile("/proc/self/mountinfo");
  if (!mountinfo) {
    return {};
  }
  std::error_code ec;
  const fs::path target = fs::weakly_canonical(fs::path(path), ec);
  const std::string wanted = ec ? path : target.string();

  std::string best;
  std::size_t bestLength = 0;
  for (const auto& line : sysfs::split(*mountinfo, "\n")) {
    const auto fields = sysfs::split(line, " ");
    if (fields.size() < 5) {
      continue;
    }
    const std::string& mountPoint = fields[4];
    if (mountPoint != wanted && mountPoint != path) {
      continue;
    }
    // Field 10 is the separator before the optional fields, so the source
    // device is the first entry after it.
    std::string device;
    if (fields.size() > 10) {
      device = fields[10];
    } else {
      for (std::size_t i = 10; i < fields.size(); ++i) {
        if (fields[i] == "-" && i + 1 < fields.size()) {
          device = fields[i + 1];
          break;
        }
      }
    }
    if (mountPoint.size() >= bestLength) {
      bestLength = mountPoint.size();
      best = device;
    }
  }
  return best;
}

DiskSnapshot DiskMonitor::sample(const std::string& path) {
  DiskSnapshot snapshot;
  snapshot.mountPoint = path;

  struct statvfs stats {};
  if (::statvfs(path.c_str(), &stats) != 0) {
    return snapshot;
  }

  const std::uint64_t blockSize = stats.f_frsize > 0 ? stats.f_frsize : stats.f_bsize;
  snapshot.totalBytes = static_cast<std::uint64_t>(stats.f_blocks) * blockSize;
  snapshot.freeBytes = static_cast<std::uint64_t>(stats.f_bfree) * blockSize;
  const std::uint64_t available =
      static_cast<std::uint64_t>(stats.f_bavail) * blockSize;
  snapshot.usedBytes = snapshot.totalBytes - available;
  snapshot.usagePercent =
      snapshot.totalBytes > 0
          ? std::clamp(static_cast<double>(snapshot.usedBytes) /
                               static_cast<double>(snapshot.totalBytes) * 100.0,
                       0.0, 100.0)
          : 0.0;
  snapshot.device = resolveDevice(path);
  snapshot.available = true;
  return snapshot;
}

}  // namespace visionai
