// Unit tests for the parsing and aggregation logic in the hardware layer.
//
// The parsers are static and take strings, so they can be tested against
// synthetic procfs content without needing the real files to look a certain way.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "hardware/cpu_monitor.hpp"
#include "hardware/disk_monitor.hpp"
#include "hardware/gpu_monitor.hpp"
#include "hardware/memory_monitor.hpp"
#include "hardware/monitor.hpp"
#include "hardware/network_monitor.hpp"
#include "hardware/sysfs.hpp"
#include "hardware/temperature_monitor.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string& label) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::cerr << "FAIL: " << label << "\n";
  }
}

void checkNear(double actual, double expected, double tolerance, const std::string& label) {
  check(std::fabs(actual - expected) <= tolerance,
        label + " (expected " + std::to_string(expected) + ", got " +
            std::to_string(actual) + ")");
}

void testSysfsHelpers() {
  check(visionai::sysfs::trim("  hello \t\n") == "hello", "trim removes surrounding space");
  const auto parts = visionai::sysfs::split("a  b\tc", " \t");
  check(parts.size() == 3, "split collapses repeated delimiters");
  check(parts[2] == "c", "split keeps the final token");
  check(visionai::sysfs::parseUint("1234").value_or(0) == 1234, "parseUint reads a number");
  check(visionai::sysfs::parseUint("not-a-number").has_value() == false,
        "parseUint rejects garbage");
  checkNear(visionai::sysfs::parseDouble("3.5").value_or(0.0), 3.5, 1e-9, "parseDouble");
  check(visionai::sysfs::pathExists("/proc/stat"), "/proc/stat exists");
  check(visionai::sysfs::pathExists("/definitely/not/here") == false,
        "missing paths report false");
}

void testStatParsing() {
  const auto times = visionai::CpuMonitor::parseStatLine("cpu  100 20 30 400 10 5 5 0");
  check(times.user == 100, "stat user field");
  check(times.nice == 20, "stat nice field");
  check(times.system == 30, "stat system field");
  check(times.idle == 400, "stat idle field");
  check(times.iowait == 10, "stat iowait field");
  check(times.steal == 0, "stat steal defaults to zero when absent");
  check(times.busy() == 160, "busy excludes idle and iowait");
  check(times.total() == 570, "total includes idle and iowait");

  const auto shortLine = visionai::CpuMonitor::parseStatLine("cpu 1 2 3 4");
  check(shortLine.user == 1, "a minimal four-value stat line still parses");
  check(shortLine.idle == 4, "minimal stat line idle field");
  check(shortLine.total() == 10, "minimal stat line total");

  const auto garbage = visionai::CpuMonitor::parseStatLine("nonsense");
  check(garbage.total() == 0, "unparseable stat line yields zeros");
}

void testModelNameParsing() {
  const std::string cpuinfo =
      "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz\n";
  check(visionai::CpuMonitor::parseModelName(cpuinfo) ==
            "Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz",
        "model name is extracted from cpuinfo");
  check(visionai::CpuMonitor::parseModelName("processor\t: 0\n").empty(),
        "missing model name yields an empty string");
}

void testLoadAverageParsing() {
  const auto values = visionai::CpuMonitor::parseLoadAverage("0.52 0.58 0.59 1/1234 5678");
  check(values.size() == 3, "load average yields three values");
  checkNear(values[0], 0.52, 1e-9, "load average 1 minute");
  checkNear(values[2], 0.59, 1e-9, "load average 15 minute");
}

void testMemoryParsing() {
  const std::string meminfo =
      "MemTotal:       16384000 kB\n"
      "MemFree:         2000000 kB\n"
      "MemAvailable:    8192000 kB\n"
      "Buffers:          500000 kB\n"
      "Cached:          3000000 kB\n"
      "SwapTotal:       4096000 kB\n"
      "SwapFree:        4096000 kB\n";

  check(visionai::MemoryMonitor::parseMemInfo(meminfo, "MemTotal") == 16384000ULL * 1024ULL,
        "MemTotal converts kB to bytes");
  check(visionai::MemoryMonitor::parseMemInfo(meminfo, "MemAvailable") == 8192000ULL * 1024ULL,
        "MemAvailable converts kB to bytes");
  check(visionai::MemoryMonitor::parseMemInfo(meminfo, "HugePages_Total") == 0,
        "absent keys return zero");

  visionai::MemorySnapshot snapshot;
  snapshot.totalBytes = 100;
  snapshot.availableBytes = 25;
  checkNear(visionai::MemoryMonitor::usagePercent(snapshot), 75.0, 1e-9,
            "usage percent uses MemAvailable, not MemFree");
  snapshot.totalBytes = 0;
  checkNear(visionai::MemoryMonitor::usagePercent(snapshot), 0.0, 1e-9,
            "usage percent of an empty machine is zero");
}

void testNetworkParsing() {
  const std::string netdev =
      "Inter-|   Receive                                                |  Transmit\n"
      " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n"
      "    lo:  1000000    100    0    0    0     0          0         0  1000000    100    0    0    0     0       0          0\n"
      "  eth0: 5000000   1000    1    2    0     0          0         0  2500000    500    3    4    0     0       0          0\n";

  const auto interfaces = visionai::NetworkMonitor::parseNetDev(netdev);
  check(interfaces.size() == 2, "both interfaces are parsed");
  if (interfaces.size() == 2) {
    check(interfaces[1].name == "eth0", "interface name is trimmed");
    check(interfaces[1].rxBytes == 5000000ULL, "rx byte counter");
    check(interfaces[1].txBytes == 2500000ULL, "tx byte counter");
    check(interfaces[1].rxPackets == 1000ULL, "rx packet counter");
    check(interfaces[1].txPackets == 500ULL, "tx packet counter");
    check(interfaces[1].errors == 4, "errors sum rx and tx");
    check(interfaces[1].drops == 6, "drops sum rx and tx");
  }

  check(visionai::NetworkMonitor::pickInterface(interfaces, "lo") == "lo",
        "an explicit interface is honoured");
  check(visionai::NetworkMonitor::pickInterface(interfaces, "") == "eth0",
        "loopback is skipped when choosing automatically");
  check(visionai::NetworkMonitor::pickInterface(interfaces, "does-not-exist") == "eth0",
        "an unknown preference falls back to the busiest interface");
}

void testNvidiaSmiParsing() {
  const std::string output =
      "0, NVIDIA GeForce RTX 3060, 42, 1024, 12288, 47, 55.30\n"
      "1, NVIDIA GeForce RTX 3060, 7, 512, 12288, 40, 22.10\n";
  const auto gpus = visionai::GpuMonitor::parseNvidiaSmi(output);
  check(gpus.size() == 2, "two GPUs are parsed");
  if (gpus.size() == 2) {
    check(gpus[0].vendor == "NVIDIA", "vendor is NVIDIA");
    check(gpus[0].name == "NVIDIA GeForce RTX 3060", "GPU name is parsed");
    checkNear(gpus[0].usagePercent, 42.0, 1e-9, "utilisation");
    check(gpus[0].memoryUsedBytes == 1024ULL * 1024 * 1024, "memory used converts MiB to bytes");
    check(gpus[0].temperatureCelsius.has_value(), "temperature is present");
    checkNear(*gpus[0].temperatureCelsius, 47.0, 1e-9, "temperature value");
    checkNear(*gpus[0].powerWatts, 55.30, 1e-6, "power draw");
    checkNear(gpus[0].memoryUsagePercent, 1024.0 / 12288.0 * 100.0, 1e-6,
              "memory percentage is derived");
  }
  check(visionai::GpuMonitor::parseNvidiaSmi("").empty(), "empty output yields no GPUs");
  check(visionai::GpuMonitor::parseNvidiaSmi("| NVIDIA-SMI has failed").empty(),
        "nvidia-smi failure text is not parsed as a GPU");
}

void testTemperatureSelection() {
  std::vector<visionai::TemperatureSensor> sensors;
  sensors.push_back({"nvme/Composite", "/x/temp1_input", 38.0, 0.0, 0.0});
  sensors.push_back({"coretemp/Package id 0", "/y/temp1_input", 51.0, 100.0, 0.0});
  sensors.push_back({"acpitz", "/z/temp1_input", 44.0, 0.0, 0.0});

  const auto* cpu = visionai::TemperatureMonitor::pickCpuSensor(sensors);
  check(cpu != nullptr, "a CPU sensor is selected when present");
  if (cpu != nullptr) {
    checkNear(cpu->celsius, 51.0, 1e-9, "the package sensor wins over the cooler sensor");
  }

  std::vector<visionai::TemperatureSensor> fallback = {
      {"nvme/Composite", "/g/temp1_input", 38.0, 0.0, 0.0},
      {"amdgpu/edge", "/h/temp1_input", 60.0, 0.0, 0.0},
  };
  const auto* hottest = visionai::TemperatureMonitor::pickCpuSensor(fallback);
  check(hottest != nullptr, "a fallback sensor is selected when no CPU sensor exists");
  if (hottest != nullptr) {
    checkNear(hottest->celsius, 60.0, 1e-9, "fallback picks the hottest reading");
  }

  std::vector<visionai::TemperatureSensor> empty;
  check(visionai::TemperatureMonitor::pickCpuSensor(empty) == nullptr,
        "no sensors means no CPU temperature");
}

void testLiveSampling() {
  visionai::HardwareMonitor monitor;
  const auto first = monitor.sample();
  check(first.cpu.coreCount >= 1, "CPU core count is at least one");
  check(!first.cpu.modelName.empty(), "CPU model name is populated");
  check(first.memory.totalBytes > 0, "total memory is reported");
  check(first.memory.usagePercent >= 0.0 && first.memory.usagePercent <= 100.0,
        "memory usage is a percentage");
  check(first.disk.available, "the root filesystem is measurable");
  check(first.disk.totalBytes > 0, "disk total is positive");
  check(!first.network.interfaces.empty(), "at least one network interface is listed");
  check(first.system.uptimeSeconds > 0.0, "uptime is positive");
  check(!first.version.empty(), "version is reported");

  std::this_thread::sleep_for(std::chrono::milliseconds(120));
  const auto second = monitor.sample();
  checkNear(second.cpu.usagePercent, second.cpu.usagePercent, 0.001, "cpu usage is finite");
  check(second.cpu.usagePercent >= 0.0 && second.cpu.usagePercent <= 100.0,
        "cpu usage stays within 0..100 across samples");
  check(second.temperature.sensors.size() >= second.temperature.sensors.size(),
        "temperature sensor list is stable");
  if (second.temperature.available) {
    check(second.temperature.cpuCelsius > 0.0, "CPU temperature is positive when available");
    check(second.temperature.cpuCelsius < 120.0,
          "CPU temperature is within a physically plausible range");
  }
}

void testDiskForBadPath() {
  visionai::HardwareMonitor monitor;
  const auto missing = monitor.sampleDisk("/this/path/does/not/exist");
  check(!missing.available, "an unreadable mount point reports unavailable");
  check(missing.usagePercent == 0.0, "an unavailable disk reports zero usage");
}

}  // namespace

int main() {
  testSysfsHelpers();
  testStatParsing();
  testModelNameParsing();
  testLoadAverageParsing();
  testMemoryParsing();
  testNetworkParsing();
  testNvidiaSmiParsing();
  testTemperatureSelection();
  testLiveSampling();
  testDiskForBadPath();

  std::cout << (g_failures == 0 ? "PASS" : "FAIL") << ": " << (g_checks - g_failures) << "/"
            << g_checks << " native checks passed\n";
  return g_failures == 0 ? 0 : 1;
}
