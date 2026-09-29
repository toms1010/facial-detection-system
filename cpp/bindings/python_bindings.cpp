// pybind11 bindings: the primary Python <-> C++ interface.
//
// Every snapshot is converted into plain Python dicts / lists / floats so the
// Python side never has to know the C++ struct layout.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "hardware/monitor.hpp"

namespace py = pybind11;
using visionai::CpuSnapshot;
using visionai::DiskSnapshot;
using visionai::GpuSnapshot;
using visionai::HardwareMonitor;
using visionai::HardwareSnapshot;
using visionai::MemorySnapshot;
using visionai::NetworkSnapshot;
using visionai::SystemSnapshot;
using visionai::TemperatureSnapshot;

namespace {

py::dict cpuToDict(const CpuSnapshot& cpu) {
  py::list cores;
  for (const auto& core : cpu.cores) {
    py::dict entry;
    entry["index"] = core.index;
    entry["usage_percent"] = core.usagePercent;
    entry["frequency_mhz"] = core.frequencyMhz;
    cores.append(entry);
  }
  py::dict out;
  out["usage_percent"] = cpu.usagePercent;
  out["user_percent"] = cpu.userPercent;
  out["system_percent"] = cpu.systemPercent;
  out["idle_percent"] = cpu.idlePercent;
  out["iowait_percent"] = cpu.iowaitPercent;
  out["frequency_mhz"] = cpu.frequencyMhz;
  out["max_frequency_mhz"] = cpu.maxFrequencyMhz;
  out["load_average_1"] = cpu.loadAverage1;
  out["load_average_5"] = cpu.loadAverage5;
  out["load_average_15"] = cpu.loadAverage15;
  out["core_count"] = cpu.coreCount;
  out["model_name"] = cpu.modelName;
  out["cores"] = cores;
  return out;
}

py::dict memoryToDict(const MemorySnapshot& memory) {
  py::dict out;
  out["total_bytes"] = memory.totalBytes;
  out["used_bytes"] = memory.usedBytes;
  out["available_bytes"] = memory.availableBytes;
  out["free_bytes"] = memory.freeBytes;
  out["cached_bytes"] = memory.cachedBytes;
  out["buffers_bytes"] = memory.buffersBytes;
  out["swap_total_bytes"] = memory.swapTotalBytes;
  out["swap_free_bytes"] = memory.swapFreeBytes;
  out["usage_percent"] = memory.usagePercent;
  out["swap_usage_percent"] = memory.swapUsagePercent;
  return out;
}

py::dict temperatureToDict(const TemperatureSnapshot& temperature) {
  py::list sensors;
  for (const auto& sensor : temperature.sensors) {
    py::dict entry;
    entry["label"] = sensor.label;
    entry["path"] = sensor.path;
    entry["celsius"] = sensor.celsius;
    entry["critical_celsius"] = sensor.criticalCelsius;
    entry["max_celsius"] = sensor.maxCelsius;
    sensors.append(entry);
  }
  py::dict out;
  out["available"] = temperature.available;
  out["cpu_celsius"] = temperature.cpuCelsius;
  out["max_celsius"] = temperature.maxCelsius;
  out["cpu_source"] = temperature.cpuSource;
  out["sensors"] = sensors;
  return out;
}

py::dict diskToDict(const DiskSnapshot& disk) {
  py::dict out;
  out["available"] = disk.available;
  out["mount_point"] = disk.mountPoint;
  out["device"] = disk.device;
  out["filesystem"] = disk.filesystem;
  out["total_bytes"] = disk.totalBytes;
  out["used_bytes"] = disk.usedBytes;
  out["free_bytes"] = disk.freeBytes;
  out["usage_percent"] = disk.usagePercent;
  return out;
}

py::dict networkToDict(const NetworkSnapshot& network) {
  py::list interfaces;
  for (const auto& iface : network.interfaces) {
    py::dict entry;
    entry["name"] = iface.name;
    entry["rx_bytes"] = iface.rxBytes;
    entry["tx_bytes"] = iface.txBytes;
    entry["rx_bytes_per_second"] = iface.rxBytesPerSecond;
    entry["tx_bytes_per_second"] = iface.txBytesPerSecond;
    entry["rx_packets_per_second"] = iface.rxPacketsPerSecond;
    entry["tx_packets_per_second"] = iface.txPacketsPerSecond;
    entry["errors"] = iface.errors;
    entry["drops"] = iface.drops;
    entry["is_up"] = iface.isUp;
    interfaces.append(entry);
  }
  py::dict out;
  out["primary_interface"] = network.primaryInterface;
  out["rx_bytes_per_second"] = network.rxBytesPerSecond;
  out["tx_bytes_per_second"] = network.txBytesPerSecond;
  out["interfaces"] = interfaces;
  return out;
}

py::list gpusToList(const std::vector<GpuSnapshot>& gpus) {
  py::list out;
  for (const auto& gpu : gpus) {
    py::dict entry;
    entry["present"] = gpu.present;
    entry["vendor"] = gpu.vendor;
    entry["name"] = gpu.name;
    entry["usage_percent"] = gpu.usagePercent;
    entry["memory_usage_percent"] = gpu.memoryUsagePercent;
    entry["memory_used_bytes"] = gpu.memoryUsedBytes;
    entry["memory_total_bytes"] = gpu.memoryTotalBytes;
    entry["index"] = gpu.index;
    entry["source"] = gpu.source;
    entry["detail"] = gpu.detail;
    if (gpu.temperatureCelsius.has_value()) {
      entry["temperature_celsius"] = *gpu.temperatureCelsius;
    } else {
      entry["temperature_celsius"] = py::none();
    }
    if (gpu.powerWatts.has_value()) {
      entry["power_watts"] = *gpu.powerWatts;
    } else {
      entry["power_watts"] = py::none();
    }
    out.append(entry);
  }
  return out;
}

py::dict systemToDict(const SystemSnapshot& system) {
  py::dict out;
  out["hostname"] = system.hostname;
  out["kernel"] = system.kernel;
  out["distribution"] = system.distribution;
  out["uptime_seconds"] = system.uptimeSeconds;
  out["process_count"] = system.processCount;
  out["architecture"] = system.architecture;
  return out;
}

py::dict toDict(const HardwareSnapshot& s) {
  py::dict out;
  out["cpu"] = cpuToDict(s.cpu);
  out["memory"] = memoryToDict(s.memory);
  out["temperature"] = temperatureToDict(s.temperature);
  out["disk"] = diskToDict(s.disk);
  out["network"] = networkToDict(s.network);
  out["gpus"] = gpusToList(s.gpus);
  out["system"] = systemToDict(s.system);
  out["sample_seconds"] = s.sampleSeconds;
  out["version"] = s.version;
  out["backend"] = "pybind11";
  return out;
}

}  // namespace

PYBIND11_MODULE(_visionai_native, m) {
  m.doc() = "Native Linux hardware monitoring for linux-ai-vision";
  m.attr("__version__") = HardwareMonitor::kVersion;

  m.def("capabilities", &HardwareMonitor::capabilities,
        "Comma separated list of supported telemetry sources.");
  m.def("library_version", &HardwareMonitor::version, "Version of the native layer.");

  m.def(
      "parse_stat_line",
      [](const std::string& line) {
        const auto times = visionai::CpuMonitor::parseStatLine(line);
        py::dict d;
        d["user"] = times.user;
        d["nice"] = times.nice;
        d["system"] = times.system;
        d["idle"] = times.idle;
        d["iowait"] = times.iowait;
        d["irq"] = times.irq;
        d["softirq"] = times.softirq;
        d["steal"] = times.steal;
        return d;
      },
      py::arg("line"), "Parse one /proc/stat CPU line into its counters.");

  m.def(
      "parse_meminfo",
      [](const std::string& content, const std::string& key) {
        return visionai::MemoryMonitor::parseMemInfo(content, key);
      },
      py::arg("content"), py::arg("key"),
      "Extract one /proc/meminfo field in bytes.");

  m.def(
      "parse_net_dev",
      [](const std::string& content) {
        NetworkSnapshot parsed;
        parsed.interfaces = visionai::NetworkMonitor::parseNetDev(content);
        return networkToDict(parsed)["interfaces"].cast<py::list>();
      },
      py::arg("content"), "Parse /proc/net/dev content into interface counters.");

  m.def(
      "parse_nvidia_smi",
      [](const std::string& output) {
        return gpusToList(visionai::GpuMonitor::parseNvidiaSmi(output));
      },
      py::arg("output"), "Parse nvidia-smi CSV output.");

  py::class_<HardwareMonitor>(m, "HardwareMonitor")
      .def(py::init<>())
      .def(
          "snapshot",
          [](HardwareMonitor& self) { return toDict(self.sample()); },
          "Sample every subsystem at once.")
      .def(
          "sample_cpu", [](HardwareMonitor& self) { return cpuToDict(self.sampleCpu()); },
          "Sample CPU statistics.")
      .def(
          "sample_memory", [](HardwareMonitor& self) { return memoryToDict(self.sampleMemory()); },
          "Sample memory statistics.")
      .def(
          "sample_temperature",
          [](HardwareMonitor& self) { return temperatureToDict(self.sampleTemperature()); },
          "Sample temperature sensors.")
      .def(
          "sample_disk",
          [](HardwareMonitor& self, const std::string& path) {
            return diskToDict(self.sampleDisk(path));
          },
          py::arg("path") = "/", "Sample disk usage for a mount point.")
      .def(
          "sample_gpus", [](HardwareMonitor& self) { return gpusToList(self.sampleGpus()); },
          "Sample GPU telemetry.")
      .def(
          "sample_network",
          [](HardwareMonitor& self, const std::string& interfaceName) {
            return networkToDict(self.sampleNetwork(interfaceName));
          },
          py::arg("interface") = "", "Sample network throughput.")
      .def(
          "sample_system",
          [](HardwareMonitor&) { return systemToDict(HardwareMonitor::sampleSystem()); },
          "Sample host information.")
      .def("set_disk_path", &HardwareMonitor::setDiskPath, py::arg("path"))
      .def("set_network_interface", &HardwareMonitor::setNetworkInterface, py::arg("name"));

  m.def(
      "sample_all",
      [](const std::string& diskPath, const std::string& interfaceName) {
        HardwareMonitor monitor;
        monitor.setDiskPath(diskPath.empty() ? "/" : diskPath);
        monitor.setNetworkInterface(interfaceName);
        return toDict(monitor.sample());
      },
      py::arg("disk_path") = "/", py::arg("network_interface") = "",
      "One-shot helper returning a full snapshot dictionary.");
}
