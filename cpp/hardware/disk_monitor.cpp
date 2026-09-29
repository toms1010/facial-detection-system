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

std::string DiskMonitor::resolveDevice(const std::string& path, std::string* filesystem) {
  const auto mountinfo = sysfs::readFile("/proc/self/mountinfo");
  if (!mountinfo) {
    return {};
  }
  std::error_code ec;
  const fs::path target = fs::weakly_canonical(fs::path(path), ec);
  const std::string wanted = ec ? path : target.string();

  std::string best;
  bool found = false;
  for (const auto& line : sysfs::split(*mountinfo, "\n")) {
    const auto fields = sysfs::split(line, " ");
    if (fields.size() < 5) {
      continue;
    }
    const std::string& mountPoint = fields[4];
    if (mountPoint != wanted && mountPoint != path) {
      continue;
    }
    // /proc/self/mountinfo layout:
    //   0 id, 1 parent, 2 major:minor, 3 root, 4 mount point, 5 options,
    //   then zero or more optional fields terminated by "-", then
    //   fs type, mount source, super options.
    // The separator's index therefore moves with the number of optional
    // fields, so the source is always two entries after it.
    for (std::size_t i = 6; i + 2 < fields.size(); ++i) {
      if (fields[i] != "-") {
        continue;
      }
      // The first matching line is the topmost mount; later ones are stacked
      // underneath it.
      if (!found) {
        found = true;
        best = fields[i + 2];
        if (filesystem != nullptr) {
          *filesystem = fields[i + 1];
        }
      }
      break;
    }
    if (found) {
      break;
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
  snapshot.filesystem = {};
  snapshot.device = resolveDevice(path, &snapshot.filesystem);
  snapshot.available = true;
  return snapshot;
}

}  // namespace visionai
