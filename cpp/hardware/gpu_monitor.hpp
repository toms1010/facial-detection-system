// GPU discovery and utilisation across vendors.
//
// NVIDIA: /proc/driver/nvidia/gpus/*/ (temperature, power) and the
//         utilisation_gpu / memory fields exported by the driver.
// AMD:    /sys/class/drm/card*/device/{gpu_busy_percent, mem_info_vram_*}
// Intel:  /sys/class/drm/card*/gt_act_freq_mhz and friends; i915 often
//         exposes no busy percentage, in which case it is reported as absent.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "types.hpp"

namespace visionai {

class GpuMonitor {
 public:
  void discover();
  std::vector<GpuSnapshot> sample();

  static std::optional<std::string> readNvidiaSmi();
  static std::vector<GpuSnapshot> parseNvidiaSmi(const std::string& output);
  static std::vector<std::string> listDrmCards();

  // True when an NVIDIA driver is present, checked without spawning anything.
  static bool nvidiaDriverPresent();
  static bool nvidiaSmiOnPath();

  const std::vector<GpuSnapshot>& last() const { return last_; }

 private:
  std::vector<std::string> cards_;
  bool discovered_ = false;
  bool nvidiaUsable_ = false;
  bool nvidiaProbed_ = false;
  std::vector<GpuSnapshot> last_;
};

}  // namespace visionai
