#include "network_monitor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "sysfs.hpp"

namespace visionai {

std::vector<NetworkInterface> NetworkMonitor::parseNetDev(const std::string& content) {
  std::vector<NetworkInterface> interfaces;
  bool inHeader = false;
  for (const auto& line : sysfs::split(content, "\n")) {
    const std::string trimmed = sysfs::trim(line);
    if (trimmed.empty()) {
      continue;
    }
    if (trimmed.rfind("Inter-", 0) == 0) {
      inHeader = true;
      continue;
    }
    if (inHeader) {
      inHeader = false;
      continue;
    }
    const auto colon = trimmed.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string name = sysfs::trim(std::string_view(trimmed).substr(0, colon));
    if (name.empty()) {
      continue;
    }
    const auto fields =
        sysfs::split(std::string_view(trimmed).substr(colon + 1), " \t");
    if (fields.size() < 16) {
      continue;
    }
    NetworkInterface iface;
    iface.name = name;
    iface.rxBytes = sysfs::parseUint(fields[0]).value_or(0);
    iface.rxPackets = sysfs::parseUint(fields[1]).value_or(0);
    iface.txBytes = sysfs::parseUint(fields[8]).value_or(0);
    iface.txPackets = sysfs::parseUint(fields[9]).value_or(0);
    iface.errors = static_cast<int>(sysfs::parseUint(fields[2]).value_or(0) +
                                    sysfs::parseUint(fields[10]).value_or(0));
    iface.drops = static_cast<int>(sysfs::parseUint(fields[3]).value_or(0) +
                                   sysfs::parseUint(fields[11]).value_or(0));
    const auto operstate = sysfs::readFirstLine("/sys/class/net/" + name + "/operstate");
    iface.isUp = operstate && *operstate != "down";
    interfaces.push_back(iface);
  }
  return interfaces;
}

std::string NetworkMonitor::pickInterface(
    const std::vector<NetworkInterface>& interfaces, const std::string& preferred) {
  if (!preferred.empty()) {
    for (const auto& iface : interfaces) {
      if (iface.name == preferred) {
        return preferred;
      }
    }
  }
  std::string best;
  std::uint64_t bestTraffic = 0;
  for (const auto& iface : interfaces) {
    if (iface.name == "lo") {
      continue;
    }
    const std::uint64_t traffic = iface.rxBytes + iface.txBytes;
    if (traffic > bestTraffic) {
      bestTraffic = traffic;
      best = iface.name;
    }
  }
  return best.empty() && !interfaces.empty() ? interfaces.front().name : best;
}

NetworkSnapshot NetworkMonitor::sample(const std::string& preferredInterface) {
  NetworkSnapshot snapshot;
  const auto content = sysfs::readFile("/proc/net/dev");
  if (!content) {
    return snapshot;
  }

  auto interfaces = parseNetDev(*content);
  const auto now = std::chrono::steady_clock::now();
  const double elapsed =
      primed_ ? std::chrono::duration<double>(now - lastSample_).count() : 0.0;

  if (primed_ && elapsed > 1e-6) {
    for (auto& iface : interfaces) {
      const auto it = previous_.find(iface.name);
      if (it == previous_.end()) {
        continue;
      }
      const auto delta = [](std::uint64_t current, std::uint64_t old) -> double {
        return current >= old ? static_cast<double>(current - old) : 0.0;
      };
      iface.rxBytesPerSecond = delta(iface.rxBytes, it->second.rxBytes) / elapsed;
      iface.txBytesPerSecond = delta(iface.txBytes, it->second.txBytes) / elapsed;
      iface.rxPacketsPerSecond = delta(iface.rxPackets, it->second.rxPackets) / elapsed;
      iface.txPacketsPerSecond = delta(iface.txPackets, it->second.txPackets) / elapsed;
    }
  }

  snapshot.primaryInterface = pickInterface(interfaces, preferredInterface);
  for (const auto& iface : interfaces) {
    if (iface.name == snapshot.primaryInterface) {
      snapshot.rxBytesPerSecond = iface.rxBytesPerSecond;
      snapshot.txBytesPerSecond = iface.txBytesPerSecond;
      break;
    }
  }

  previous_.clear();
  for (const auto& iface : interfaces) {
    previous_[iface.name] = {iface.rxBytes, iface.txBytes, iface.rxPackets, iface.txPackets};
  }
  lastSample_ = now;
  primed_ = true;
  snapshot.interfaces = std::move(interfaces);
  return snapshot;
}

}  // namespace visionai
