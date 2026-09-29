// Aggregate hardware monitor combining every subsystem.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cpu_monitor.hpp"
#include "disk_monitor.hpp"
#include "gpu_monitor.hpp"
#include "memory_monitor.hpp"
#include "network_monitor.hpp"
#include "temperature_monitor.hpp"
#include "types.hpp"

namespace visionai {

struct HardwareSnapshot {
  CpuSnapshot cpu;
  MemorySnapshot memory;
  TemperatureSnapshot temperature;
  DiskSnapshot disk;
  NetworkSnapshot network;
  SystemSnapshot system;
  std::vector<GpuSnapshot> gpus;
  double sampleSeconds = 0.0;
  std::string version;
};

class HardwareMonitor {
 public:
  HardwareMonitor();

  // Samples every subsystem. The first call primes CPU and network deltas.
  HardwareSnapshot sample();

  // A single subsystem, for callers that only need one number per tick.
  CpuSnapshot sampleCpu();
  MemorySnapshot sampleMemory();
  TemperatureSnapshot sampleTemperature();
  DiskSnapshot sampleDisk(const std::string& path = "/");
  std::vector<GpuSnapshot> sampleGpus();
  NetworkSnapshot sampleNetwork(const std::string& preferred = "");

  static SystemSnapshot sampleSystem();
  static std::string version();
  static std::string capabilities();

  void setDiskPath(const std::string& path) { diskPath_ = path; }
  void setNetworkInterface(const std::string& name) { networkInterface_ = name; }

  static constexpr const char* kVersion = "1.0.0";

 private:
  CpuMonitor cpu_;
  MemoryMonitor memory_;
  TemperatureMonitor temperature_;
  DiskMonitor disk_;
  GpuMonitor gpu_;
  NetworkMonitor network_;
  std::string diskPath_ = "/";
  std::string networkInterface_;
};

}  // namespace visionai
