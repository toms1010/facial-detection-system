#include "gpu_monitor.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>

#include "sysfs.hpp"

namespace visionai {
namespace {

constexpr const char* kNvidiaSmi =
    "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
    "temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null";

std::string vendorFromDriver(const std::string& driverLink) {
  if (driverLink.find("amdgpu") != std::string::npos) {
    return "AMD";
  }
  if (driverLink.find("i915") != std::string::npos) {
    return "Intel";
  }
  if (driverLink.find("nvidia") != std::string::npos) {
    return "NVIDIA";
  }
  return "Unknown";
}

std::optional<double> readDouble(const std::string& path) {
  const auto value = sysfs::readFirstLine(path);
  if (!value) {
    return std::nullopt;
  }
  return sysfs::parseDouble(*value);
}

std::optional<std::uint64_t> readUint(const std::string& path) {
  const auto value = sysfs::readFirstLine(path);
  if (!value) {
    return std::nullopt;
  }
  return sysfs::parseUint(*value);
}

// PCI device ids live in the uevent file as PCI_ID=VVVV:DDDD. Resolving them to
// marketing names needs a PCI database, which is not worth a dependency here,
// so the driver plus vendor is the honest description.
std::string gpuName(const std::string& devicePath, const std::string& vendor,
                    const std::string& driverName) {
  if (const auto label = sysfs::readFirstLine(devicePath + "/label")) {
    if (!label->empty() && label->find('=') == std::string::npos) {
      return *label;
    }
  }
  const std::string vendorPart = vendor == "Unknown" ? "GPU" : vendor + " GPU";
  if (!driverName.empty()) {
    return vendorPart + " (" + driverName + ")";
  }
  return vendorPart;
}

}  // namespace

std::optional<std::string> GpuMonitor::readNvidiaSmi() {
  std::array<char, 4096> buffer{};
  std::FILE* pipe = ::popen(kNvidiaSmi, "r");
  if (pipe == nullptr) {
    return std::nullopt;
  }
  std::string output;
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  const int status = ::pclose(pipe);
  if (status != 0 || output.find("NVIDIA-SMI") != std::string::npos) {
    return std::nullopt;
  }
  return output;
}

std::vector<GpuSnapshot> GpuMonitor::parseNvidiaSmi(const std::string& output) {
  std::vector<GpuSnapshot> gpus;
  int index = 0;
  for (const auto& line : sysfs::split(output, "\n")) {
    const std::string trimmed = sysfs::trim(line);
    if (trimmed.empty()) {
      continue;
    }
    const auto fields = sysfs::split(trimmed, ",");
    if (fields.size() < 6) {
      continue;
    }
    GpuSnapshot gpu;
    gpu.present = true;
    gpu.vendor = "NVIDIA";
    gpu.index = index++;
    gpu.source = "nvidia-smi";
    gpu.name = sysfs::trim(fields[1]);
    if (gpu.name.empty() || gpu.name == "[N/A]") {
      gpu.name = "NVIDIA GPU";
    }
    if (const auto util = sysfs::parseDouble(fields[2])) {
      gpu.usagePercent = *util;
    }
    if (const auto used = sysfs::parseUint(fields[3])) {
      gpu.memoryUsedBytes = *used * 1024ULL * 1024ULL;
    }
    if (const auto total = sysfs::parseUint(fields[4])) {
      gpu.memoryTotalBytes = *total * 1024ULL * 1024ULL;
    }
    if (const auto temp = sysfs::parseDouble(fields[5])) {
      if (*temp > 0.0) {
        gpu.temperatureCelsius = *temp;
      }
    }
    if (fields.size() > 6) {
      if (const auto power = sysfs::parseDouble(fields[6])) {
        if (*power > 0.0) {
          gpu.powerWatts = *power;
        }
      }
    }
    if (gpu.memoryTotalBytes > 0) {
      gpu.memoryUsagePercent = static_cast<double>(gpu.memoryUsedBytes) /
                               static_cast<double>(gpu.memoryTotalBytes) * 100.0;
    }
    gpus.push_back(gpu);
  }
  return gpus;
}

std::vector<std::string> GpuMonitor::listDrmCards() {
  std::vector<std::string> cards;
  for (const auto& entry : sysfs::listDirectories("/sys/class/drm")) {
    if (entry.rfind("card", 0) != 0) {
      continue;
    }
    if (entry.find('-') != std::string::npos) {
      continue;
    }
    if (sysfs::pathExists("/sys/class/drm/" + entry + "/device")) {
      cards.push_back(entry);
    }
  }
  return cards;
}

void GpuMonitor::discover() {
  cards_ = listDrmCards();
  discovered_ = true;
}

bool GpuMonitor::nvidiaDriverPresent() {
  return !sysfs::listDirectories("/proc/driver/nvidia/gpus").empty();
}

bool GpuMonitor::nvidiaSmiOnPath() {
  const auto path = std::getenv("PATH");
  if (path == nullptr) {
    return false;
  }
  for (const auto& directory : sysfs::split(path, ":")) {
    if (directory.empty()) {
      continue;
    }
    if (sysfs::pathExists(directory + "/nvidia-smi")) {
      return true;
    }
  }
  return false;
}

std::vector<GpuSnapshot> GpuMonitor::sample() {
  if (!discovered_) {
    discover();
  }

  std::vector<GpuSnapshot> gpus;

  // Spawning nvidia-smi costs hundreds of milliseconds, which is far too slow
  // for a per-second telemetry tick. Probe once, then reuse the answer.
  if (!nvidiaProbed_) {
    nvidiaProbed_ = true;
    nvidiaUsable_ = nvidiaDriverPresent() && nvidiaSmiOnPath();
  }
  if (nvidiaUsable_) {
    if (const auto output = readNvidiaSmi()) {
      gpus = parseNvidiaSmi(*output);
    }
  }

  if (gpus.empty()) {
    for (std::size_t i = 0; i < cards_.size(); ++i) {
      const std::string base = "/sys/class/drm/" + cards_[i] + "/device";
      const std::string driverName = sysfs::readLinkTarget(base + "/driver").value_or("");
      const std::string vendor = vendorFromDriver(driverName);

      GpuSnapshot gpu;
      gpu.present = true;
      gpu.vendor = vendor;
      gpu.index = static_cast<int>(i);
      gpu.source = "sysfs";
      gpu.name = gpuName(base, vendor, driverName);
      if (const auto busy = readDouble(base + "/gpu_busy_percent")) {
        gpu.usagePercent = *busy;
      }
      if (const auto used = readUint(base + "/mem_info_vram_used")) {
        gpu.memoryUsedBytes = *used;
      }
      if (const auto total = readUint(base + "/mem_info_vram_total")) {
        gpu.memoryTotalBytes = *total;
      }
      if (const auto busy = readDouble(base + "/mem_busy_percent")) {
        gpu.detail = "mem_busy " + std::to_string(static_cast<int>(*busy)) + "%";
      }
      if (const auto temp = sysfs::readTemperatureCelsius(base + "/hwmon/hwmon0/temp1_input")) {
        gpu.temperatureCelsius = *temp;
      } else if (const auto power = readDouble(base + "/power1_average")) {
        const auto microwatts = *power;
        if (microwatts > 0.0) {
          gpu.powerWatts = microwatts / 1000000.0;
        }
      }
      if (gpu.usagePercent <= 0.0 && !gpu.temperatureCelsius.has_value() &&
          gpu.memoryTotalBytes == 0) {
        gpu.detail = "no utilisation telemetry exposed by this driver";
      }
      if (gpu.memoryTotalBytes > 0) {
        gpu.memoryUsagePercent = static_cast<double>(gpu.memoryUsedBytes) /
                                 static_cast<double>(gpu.memoryTotalBytes) * 100.0;
      }
      gpus.push_back(gpu);
    }
  }

  last_ = gpus;
  return gpus;
}

}  // namespace visionai
