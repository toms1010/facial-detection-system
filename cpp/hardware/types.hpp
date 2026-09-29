// Shared snapshot structures returned to Python.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace visionai {

struct CpuTimes {
  std::uint64_t user = 0;
  std::uint64_t nice = 0;
  std::uint64_t system = 0;
  std::uint64_t idle = 0;
  std::uint64_t iowait = 0;
  std::uint64_t irq = 0;
  std::uint64_t softirq = 0;
  std::uint64_t steal = 0;

  std::uint64_t busy() const {
    return user + nice + system + irq + softirq + steal;
  }
  std::uint64_t idleAll() const { return idle + iowait; }
  std::uint64_t total() const { return busy() + idleAll(); }
};

struct CpuCore {
  int index = 0;
  double usagePercent = 0.0;
  double frequencyMhz = 0.0;
};

struct CpuSnapshot {
  double usagePercent = 0.0;
  double userPercent = 0.0;
  double systemPercent = 0.0;
  double idlePercent = 0.0;
  double iowaitPercent = 0.0;
  double frequencyMhz = 0.0;
  double maxFrequencyMhz = 0.0;
  double loadAverage1 = 0.0;
  double loadAverage5 = 0.0;
  double loadAverage15 = 0.0;
  int coreCount = 0;
  std::string modelName;
  std::vector<CpuCore> cores;
};

struct MemorySnapshot {
  std::uint64_t totalBytes = 0;
  std::uint64_t usedBytes = 0;
  std::uint64_t availableBytes = 0;
  std::uint64_t freeBytes = 0;
  std::uint64_t cachedBytes = 0;
  std::uint64_t buffersBytes = 0;
  std::uint64_t swapTotalBytes = 0;
  std::uint64_t swapFreeBytes = 0;
  double usagePercent = 0.0;
  double swapUsagePercent = 0.0;
};

struct GpuSnapshot {
  bool present = false;
  std::string vendor;
  std::string name;
  double usagePercent = 0.0;
  double memoryUsagePercent = 0.0;
  std::uint64_t memoryUsedBytes = 0;
  std::uint64_t memoryTotalBytes = 0;
  std::optional<double> temperatureCelsius;
  std::optional<double> powerWatts;
  int index = 0;
  std::string source;
  std::string detail;
};

struct TemperatureSensor {
  std::string label;
  std::string path;
  double celsius = 0.0;
  double criticalCelsius = 0.0;
  double maxCelsius = 0.0;
};

struct TemperatureSnapshot {
  bool available = false;
  double cpuCelsius = 0.0;
  double maxCelsius = 0.0;
  std::string cpuSource;
  std::vector<TemperatureSensor> sensors;
};

struct DiskSnapshot {
  bool available = false;
  std::string mountPoint;
  std::string device;
  std::string filesystem;
  std::uint64_t totalBytes = 0;
  std::uint64_t usedBytes = 0;
  std::uint64_t freeBytes = 0;
  double usagePercent = 0.0;
};

struct NetworkInterface {
  std::string name;
  std::uint64_t rxBytes = 0;
  std::uint64_t txBytes = 0;
  std::uint64_t rxPackets = 0;
  std::uint64_t txPackets = 0;
  double rxBytesPerSecond = 0.0;
  double txBytesPerSecond = 0.0;
  double rxPacketsPerSecond = 0.0;
  double txPacketsPerSecond = 0.0;
  int errors = 0;
  int drops = 0;
  bool isUp = false;
};

struct NetworkSnapshot {
  std::string primaryInterface;
  double rxBytesPerSecond = 0.0;
  double txBytesPerSecond = 0.0;
  std::vector<NetworkInterface> interfaces;
};

struct SystemSnapshot {
  std::string hostname;
  std::string kernel;
  std::string distribution;
  double uptimeSeconds = 0.0;
  int processCount = 0;
  std::string architecture;
};

}  // namespace visionai
