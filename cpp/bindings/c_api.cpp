#include "c_api.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include "hardware/monitor.hpp"

namespace {

char* duplicate(const std::string& text) {
  char* buffer = static_cast<char*>(std::malloc(text.size() + 1));
  if (buffer == nullptr) {
    return nullptr;
  }
  std::memcpy(buffer, text.c_str(), text.size() + 1);
  return buffer;
}

std::string escape(const std::string& raw) {
  std::string out;
  out.reserve(raw.size() + 8);
  for (const char c : raw) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: out += c; break;
    }
  }
  return out;
}

std::string number(double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.4f", value);
  return buffer;
}

std::string quote(const std::string& value) { return "\"" + escape(value) + "\""; }

std::string cpuJson(const visionai::CpuSnapshot& cpu) {
  std::string out = "{";
  out += "\"usage_percent\":" + number(cpu.usagePercent);
  out += ",\"user_percent\":" + number(cpu.userPercent);
  out += ",\"system_percent\":" + number(cpu.systemPercent);
  out += ",\"idle_percent\":" + number(cpu.idlePercent);
  out += ",\"iowait_percent\":" + number(cpu.iowaitPercent);
  out += ",\"frequency_mhz\":" + number(cpu.frequencyMhz);
  out += ",\"load_average_1\":" + number(cpu.loadAverage1);
  out += ",\"load_average_5\":" + number(cpu.loadAverage5);
  out += ",\"load_average_15\":" + number(cpu.loadAverage15);
  out += ",\"core_count\":" + std::to_string(cpu.coreCount);
  out += ",\"model_name\":" + quote(cpu.modelName);
  out += ",\"cores\":[";
  for (std::size_t i = 0; i < cpu.cores.size(); ++i) {
    if (i > 0) out += ",";
    out += "{\"index\":" + std::to_string(cpu.cores[i].index);
    out += ",\"usage_percent\":" + number(cpu.cores[i].usagePercent) + "}";
  }
  out += "]}";
  return out;
}

std::string memoryJson(const visionai::MemorySnapshot& memory) {
  std::string out = "{";
  out += "\"total_bytes\":" + std::to_string(memory.totalBytes);
  out += ",\"used_bytes\":" + std::to_string(memory.usedBytes);
  out += ",\"available_bytes\":" + std::to_string(memory.availableBytes);
  out += ",\"free_bytes\":" + std::to_string(memory.freeBytes);
  out += ",\"cached_bytes\":" + std::to_string(memory.cachedBytes);
  out += ",\"buffers_bytes\":" + std::to_string(memory.buffersBytes);
  out += ",\"swap_total_bytes\":" + std::to_string(memory.swapTotalBytes);
  out += ",\"swap_free_bytes\":" + std::to_string(memory.swapFreeBytes);
  out += ",\"usage_percent\":" + number(memory.usagePercent);
  out += ",\"swap_usage_percent\":" + number(memory.swapUsagePercent);
  out += "}";
  return out;
}

std::string temperatureJson(const visionai::TemperatureSnapshot& temperature) {
  std::string out = "{\"available\":";
  out += temperature.available ? "true" : "false";
  out += ",\"cpu_celsius\":" + number(temperature.cpuCelsius);
  out += ",\"max_celsius\":" + number(temperature.maxCelsius);
  out += ",\"cpu_source\":" + quote(temperature.cpuSource);
  out += ",\"sensors\":[";
  for (std::size_t i = 0; i < temperature.sensors.size(); ++i) {
    if (i > 0) out += ",";
    out += "{\"label\":" + quote(temperature.sensors[i].label);
    out += ",\"celsius\":" + number(temperature.sensors[i].celsius) + "}";
  }
  out += "]}";
  return out;
}

std::string diskJson(const visionai::DiskSnapshot& disk) {
  std::string out = "{\"available\":";
  out += disk.available ? "true" : "false";
  out += ",\"mount_point\":" + quote(disk.mountPoint);
  out += ",\"device\":" + quote(disk.device);
  out += ",\"total_bytes\":" + std::to_string(disk.totalBytes);
  out += ",\"used_bytes\":" + std::to_string(disk.usedBytes);
  out += ",\"free_bytes\":" + std::to_string(disk.freeBytes);
  out += ",\"usage_percent\":" + number(disk.usagePercent);
  out += "}";
  return out;
}

std::string networkJson(const visionai::NetworkSnapshot& network) {
  std::string out = "{";
  out += "\"primary_interface\":" + quote(network.primaryInterface);
  out += ",\"rx_bytes_per_second\":" + number(network.rxBytesPerSecond);
  out += ",\"tx_bytes_per_second\":" + number(network.txBytesPerSecond);
  out += ",\"interfaces\":[";
  for (std::size_t i = 0; i < network.interfaces.size(); ++i) {
    if (i > 0) out += ",";
    const auto& iface = network.interfaces[i];
    out += "{\"name\":" + quote(iface.name);
    out += ",\"rx_bytes\":" + std::to_string(iface.rxBytes);
    out += ",\"tx_bytes\":" + std::to_string(iface.txBytes);
    out += ",\"rx_bytes_per_second\":" + number(iface.rxBytesPerSecond);
    out += ",\"tx_bytes_per_second\":" + number(iface.txBytesPerSecond);
    out += ",\"is_up\":" + std::string(iface.isUp ? "true" : "false");
    out += "}";
  }
  out += "]}";
  return out;
}

std::string gpusJson(const std::vector<visionai::GpuSnapshot>& gpus) {
  std::string out = "[";
  for (std::size_t i = 0; i < gpus.size(); ++i) {
    if (i > 0) out += ",";
    const auto& gpu = gpus[i];
    out += "{\"vendor\":" + quote(gpu.vendor);
    out += ",\"name\":" + quote(gpu.name);
    out += ",\"usage_percent\":" + number(gpu.usagePercent);
    out += ",\"memory_usage_percent\":" + number(gpu.memoryUsagePercent);
    out += ",\"memory_used_bytes\":" + std::to_string(gpu.memoryUsedBytes);
    out += ",\"memory_total_bytes\":" + std::to_string(gpu.memoryTotalBytes);
    out += ",\"temperature_celsius\":";
    out += gpu.temperatureCelsius.has_value() ? number(*gpu.temperatureCelsius) : "null";
    out += ",\"source\":" + quote(gpu.source);
    out += "}";
  }
  out += "]";
  return out;
}

std::string systemJson(const visionai::SystemSnapshot& system) {
  std::string out = "{";
  out += "\"hostname\":" + quote(system.hostname);
  out += ",\"kernel\":" + quote(system.kernel);
  out += ",\"distribution\":" + quote(system.distribution);
  out += ",\"uptime_seconds\":" + number(system.uptimeSeconds);
  out += ",\"architecture\":" + quote(system.architecture);
  out += "}";
  return out;
}

}  // namespace

extern "C" {

char* visionai_snapshot_json(const char* disk_path, const char* network_interface) {
  try {
    visionai::HardwareMonitor monitor;
    monitor.setDiskPath(disk_path != nullptr && disk_path[0] != '\0' ? disk_path : "/");
    if (network_interface != nullptr) {
      monitor.setNetworkInterface(network_interface);
    }
    const auto snapshot = monitor.sample();
    std::string out = "{";
    out += "\"cpu\":" + cpuJson(snapshot.cpu);
    out += ",\"memory\":" + memoryJson(snapshot.memory);
    out += ",\"temperature\":" + temperatureJson(snapshot.temperature);
    out += ",\"disk\":" + diskJson(snapshot.disk);
    out += ",\"network\":" + networkJson(snapshot.network);
    out += ",\"gpus\":" + gpusJson(snapshot.gpus);
    out += ",\"system\":" + systemJson(snapshot.system);
    out += ",\"sample_seconds\":" + number(snapshot.sampleSeconds);
    out += ",\"version\":" + quote(snapshot.version);
    out += ",\"backend\":\"cabi\"}";
    return duplicate(out);
  } catch (...) {
    return duplicate("{\"error\":\"snapshot failed\"}");
  }
}

char* visionai_subsystem_json(const char* subsystem, const char* argument) {
  if (subsystem == nullptr) {
    return duplicate("{\"error\":\"no subsystem requested\"}");
  }
  try {
    visionai::HardwareMonitor monitor;
    const std::string name(subsystem);
    if (name == "cpu") return duplicate(cpuJson(monitor.sampleCpu()));
    if (name == "memory") return duplicate(memoryJson(monitor.sampleMemory()));
    if (name == "temperature") return duplicate(temperatureJson(monitor.sampleTemperature()));
    if (name == "disk") {
      return duplicate(diskJson(monitor.sampleDisk(argument != nullptr ? argument : "/")));
    }
    if (name == "network") {
      return duplicate(networkJson(monitor.sampleNetwork(argument != nullptr ? argument : "")));
    }
    if (name == "gpus") return duplicate(gpusJson(monitor.sampleGpus()));
    if (name == "system") return duplicate(systemJson(visionai::HardwareMonitor::sampleSystem()));
    return duplicate("{\"error\":\"unknown subsystem\"}");
  } catch (...) {
    return duplicate("{\"error\":\"subsystem sample failed\"}");
  }
}

char* visionai_version_json(void) {
  return duplicate(std::string("{\"version\":\"") + visionai::HardwareMonitor::kVersion + "\"}");
}

char* visionai_capabilities_json(void) {
  return duplicate(std::string("{\"capabilities\":\"") +
                   visionai::HardwareMonitor::capabilities() + "\"}");
}

void visionai_free_string(char* text) { std::free(text); }

}  // extern "C"
