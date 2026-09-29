// Network throughput from /proc/net/dev and /sys/class/net.
#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "types.hpp"

namespace visionai {

class NetworkMonitor {
 public:
  // First call establishes a baseline; rates appear from the second call on.
  NetworkSnapshot sample(const std::string& preferredInterface = "");

  static std::vector<NetworkInterface> parseNetDev(const std::string& content);

  // Chooses the interface with the most traffic, or the requested one if present.
  static std::string pickInterface(
      const std::vector<NetworkInterface>& interfaces, const std::string& preferred);

 private:
  struct Counters {
    std::uint64_t rxBytes = 0;
    std::uint64_t txBytes = 0;
    std::uint64_t rxPackets = 0;
    std::uint64_t txPackets = 0;
  };

  std::map<std::string, Counters> previous_;
  std::chrono::steady_clock::time_point lastSample_{};
  bool primed_ = false;
};

}  // namespace visionai
