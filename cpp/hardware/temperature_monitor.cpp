#include "temperature_monitor.hpp"

#include <algorithm>
#include <cmath>

#include "sysfs.hpp"

namespace visionai {
namespace {

double readTemperature(const std::string& path) {
  return sysfs::readTemperatureCelsius(path).value_or(0.0);
}

double readBound(const std::string& path) {
  const auto value = readTemperature(path);
  return value > 0.0 ? value : 0.0;
}

bool looksLikeCpuChip(const std::string& chip) {
  static const char* kCpuChips[] = {"coretemp", "k10temp", "zenpower", "cpu_thermal",
                                     "soc_thermal", "acpitz", "k8temp"};
  for (const char* candidate : kCpuChips) {
    if (chip.find(candidate) != std::string::npos) {
      return true;
    }
  }
  return false;
}

}  // namespace

void TemperatureMonitor::discover() {
  probes_.clear();
  for (const auto& hwmon : sysfs::listDirectories("/sys/class/hwmon")) {
    const std::string base = "/sys/class/hwmon/" + hwmon;
    const std::string chip =
        sysfs::readFirstLine(base + "/name").value_or(hwmon);
    for (const auto& entry : sysfs::listDirectories(base)) {
      if (entry.rfind("temp", 0) != 0) {
        continue;
      }
      if (!sysfs::pathExists(base + "/" + entry + "/temp1_input")) {
        continue;
      }
      probes_.push_back({base + "/" + entry, chip});
    }
  }
  discovered_ = true;
}

std::string TemperatureMonitor::labelFor(const std::string& hwmonPath, const std::string& chip) {
  const auto labelFile = sysfs::readFirstLine(hwmonPath + "/label");
  if (labelFile && !labelFile->empty()) {
    return *labelFile;
  }
  const auto base = hwmonPath.substr(hwmonPath.find_last_of('/') + 1);
  return chip + "/" + base;
}

const TemperatureSensor* TemperatureMonitor::pickCpuSensor(
    const std::vector<TemperatureSensor>& sensors) {
  const TemperatureSensor* best = nullptr;
  for (const auto& sensor : sensors) {
    if (sensor.celsius <= 0.0) {
      continue;
    }
    const std::string haystack = sensor.label + " " + sensor.path;
    const bool cpuRelated =
        looksLikeCpuChip(haystack) || haystack.find("x86_pkg_temp") != std::string::npos ||
        haystack.find("Package id 0") != std::string::npos ||
        haystack.find("Tctl") != std::string::npos;
    if (!cpuRelated) {
      continue;
    }
    if (best == nullptr || sensor.celsius > best->celsius) {
      best = &sensor;
    }
  }
  if (best != nullptr) {
    return best;
  }
  for (const auto& sensor : sensors) {
    if (sensor.celsius > 0.0 && (best == nullptr || sensor.celsius > best->celsius)) {
      best = &sensor;
    }
  }
  return best;
}

TemperatureSnapshot TemperatureMonitor::sample() {
  TemperatureSnapshot snapshot;
  if (!discovered_) {
    discover();
  }

  for (const auto& probe : probes_) {
    TemperatureSensor sensor;
    sensor.label = labelFor(probe.path, probe.chip);
    sensor.path = probe.path + "/temp1_input";
    sensor.celsius = readTemperature(sensor.path);
    if (sensor.celsius <= 0.0) {
      continue;
    }
    sensor.criticalCelsius = readBound(probe.path + "/temp1_crit");
    sensor.maxCelsius = readBound(probe.path + "/temp1_max");
    snapshot.sensors.push_back(sensor);
  }

  for (const auto& entry : sysfs::listDirectories("/sys/class/thermal")) {
    if (entry.rfind("thermal_zone", 0) != 0) {
      continue;
    }
    const std::string base = "/sys/class/thermal/" + entry;
    const double celsius = readTemperature(base + "/temp");
    if (celsius <= 0.0) {
      continue;
    }
    TemperatureSensor sensor;
    sensor.label = sysfs::readFirstLine(base + "/type").value_or(entry);
    sensor.path = base + "/temp";
    sensor.celsius = celsius;
    sensor.criticalCelsius = readBound(base + "/trip_point_0_temp");
    snapshot.sensors.push_back(sensor);
  }

  if (const auto* cpu = pickCpuSensor(snapshot.sensors)) {
    snapshot.available = true;
    snapshot.cpuCelsius = cpu->celsius;
    snapshot.cpuSource = cpu->label;
    for (const auto& sensor : snapshot.sensors) {
      snapshot.maxCelsius = std::max(snapshot.maxCelsius, sensor.celsius);
    }
  }
  return snapshot;
}

}  // namespace visionai
