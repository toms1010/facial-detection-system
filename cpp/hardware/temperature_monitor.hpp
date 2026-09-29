// Temperature sensors from /sys/class/hwmon and /sys/class/thermal.
#pragma once

#include <string>
#include <vector>

#include "types.hpp"

namespace visionai {

class TemperatureMonitor {
 public:
  // Enumerates hwmon devices. Safe to call when none exist.
  void discover();

  // Reads every discovered sensor plus the best CPU temperature.
  TemperatureSnapshot sample();

  // Selects the most plausible CPU temperature from a sensor list.
  static const TemperatureSensor* pickCpuSensor(const std::vector<TemperatureSensor>& sensors);

  static std::string labelFor(const std::string& hwmonPath, const std::string& chip);

 private:
  struct Probe {
    std::string path;
    std::string chip;
  };
  std::vector<Probe> probes_;
  bool discovered_ = false;
};

}  // namespace visionai
