#include "monitor.hpp"

#include <chrono>
#include <fstream>
#include <sstream>
#include <sys/utsname.h>
#include <unistd.h>

#include "sysfs.hpp"

namespace visionai {

HardwareMonitor::HardwareMonitor() {
  temperature_.discover();
  gpu_.discover();
}

CpuSnapshot HardwareMonitor::sampleCpu() { return cpu_.sample(); }
MemorySnapshot HardwareMonitor::sampleMemory() { return memory_.sample(); }
TemperatureSnapshot HardwareMonitor::sampleTemperature() { return temperature_.sample(); }
DiskSnapshot HardwareMonitor::sampleDisk(const std::string& path) { return disk_.sample(path); }
std::vector<GpuSnapshot> HardwareMonitor::sampleGpus() { return gpu_.sample(); }
NetworkSnapshot HardwareMonitor::sampleNetwork(const std::string& preferred) {
  return network_.sample(preferred.empty() ? networkInterface_ : preferred);
}

std::string HardwareMonitor::version() { return kVersion; }

std::string HardwareMonitor::capabilities() {
  std::ostringstream out;
  out << "cpu,per_core_cpu,memory,swap,temperature,hwmon,disk,network,gpu,nvidia,amdgpu,intel_gpu,system";
  return out.str();
}

SystemSnapshot HardwareMonitor::sampleSystem() {
  SystemSnapshot snapshot;
  utsname uts{};
  if (::uname(&uts) == 0) {
    snapshot.kernel = std::string(uts.sysname) + " " + uts.release;
    snapshot.architecture = uts.machine;
    snapshot.hostname = uts.nodename;
  }
  if (const auto uptime = sysfs::readFirstLine("/proc/uptime")) {
    if (const auto seconds = sysfs::parseDouble(*uptime)) {
      snapshot.uptimeSeconds = *seconds;
    }
  }
  if (const auto osRelease = sysfs::readFile("/etc/os-release")) {
    for (const auto& line : sysfs::split(*osRelease, "\n")) {
      if (line.rfind("PRETTY_NAME=", 0) == 0) {
        std::string value = line.substr(12);
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
          value = value.substr(1, value.size() - 2);
        }
        snapshot.distribution = value;
        break;
      }
    }
  }
  std::ifstream processes("/proc/loadavg");
  if (processes.is_open()) {
    std::string line;
    std::getline(processes, line);
    const auto fields = sysfs::split(line, " ");
    if (fields.size() > 3) {
      // fields[3] is "running/total"; the total is what callers want.
      const std::string& runningTotal = fields[3];
      const auto slash = runningTotal.find('/');
      const auto& target = slash == std::string::npos ? runningTotal
                                                      : runningTotal.substr(slash + 1);
      if (const auto total = sysfs::parseUint(target)) {
        snapshot.processCount = static_cast<int>(*total);
      }
    }
  }
  return snapshot;
}

HardwareSnapshot HardwareMonitor::sample() {
  const auto start = std::chrono::steady_clock::now();
  HardwareSnapshot snapshot;
  snapshot.cpu = cpu_.sample();
  snapshot.memory = memory_.sample();
  snapshot.temperature = temperature_.sample();
  snapshot.disk = disk_.sample(diskPath_);
  snapshot.network = network_.sample(networkInterface_);
  snapshot.gpus = gpu_.sample();
  snapshot.system = sampleSystem();
  snapshot.version = kVersion;
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start);
  snapshot.sampleSeconds = elapsed.count();
  return snapshot;
}

}  // namespace visionai
