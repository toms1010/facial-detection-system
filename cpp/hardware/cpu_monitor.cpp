#include "cpu_monitor.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <thread>

#include "sysfs.hpp"

namespace fs = std::filesystem;

namespace visionai {
namespace {

std::string cpuModelName() {
  if (const auto content = sysfs::readFile("/proc/cpuinfo")) {
    const auto name = CpuMonitor::parseModelName(*content);
    if (!name.empty()) {
      return name;
    }
  }
  return "Unknown CPU";
}

int cpuCoreCount() {
  const unsigned int count = std::thread::hardware_concurrency();
  return count > 0 ? static_cast<int>(count) : 1;
}

double averageFrequencyMhz() {
  double sum = 0.0;
  int seen = 0;
  for (const auto& entry : sysfs::listDirectories("/sys/devices/system/cpu")) {
    if (entry.rfind("cpu", 0) != 0) {
      continue;
    }
    const auto path = "/sys/devices/system/cpu/" + entry + "/cpufreq/cpuinfo_max_freq";
    if (const auto khz = sysfs::parseUint(sysfs::readFirstLine(path).value_or(""))) {
      sum += static_cast<double>(*khz) / 1000.0;
      ++seen;
    }
  }
  return seen > 0 ? sum / seen : 0.0;
}

double maxFrequencyMhz() {
  double best = 0.0;
  for (const auto& entry : sysfs::listDirectories("/sys/devices/system/cpu")) {
    if (entry.rfind("cpu", 0) != 0) {
      continue;
    }
    for (const char* leaf : {"cpuinfo_max_freq", "scaling_max_freq"}) {
      const auto path = "/sys/devices/system/cpu/" + entry + "/cpufreq/" + leaf;
      if (const auto khz = sysfs::parseUint(sysfs::readFirstLine(path).value_or(""))) {
        best = std::max(best, static_cast<double>(*khz) / 1000.0);
      }
    }
  }
  return best;
}

double percentage(double part, double total) {
  if (total <= 0.0) {
    return 0.0;
  }
  return std::clamp(part / total * 100.0, 0.0, 100.0);
}

}  // namespace

CpuMonitor::CpuMonitor() = default;

CpuTimes CpuMonitor::parseStatLine(const std::string& line) {
  CpuTimes times;
  const auto fields = sysfs::split(line, " \t");
  if (fields.size() < 5) {
    return times;
  }
  const auto value = [&fields](std::size_t index) -> std::uint64_t {
    return index < fields.size() ? sysfs::parseUint(fields[index]).value_or(0) : 0;
  };
  times.user = value(1);
  times.nice = value(2);
  times.system = value(3);
  times.idle = value(4);
  times.iowait = fields.size() > 5 ? value(5) : 0;
  times.irq = fields.size() > 6 ? value(6) : 0;
  times.softirq = fields.size() > 7 ? value(7) : 0;
  times.steal = fields.size() > 8 ? value(8) : 0;
  return times;
}

std::string CpuMonitor::parseModelName(const std::string& cpuinfo) {
  for (const auto& line : sysfs::split(cpuinfo, "\n")) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string key = sysfs::trim(std::string_view(line).substr(0, colon));
    if (key == "model name" || key == "Model" || key == "Processor") {
      return sysfs::trim(std::string_view(line).substr(colon + 1));
    }
  }
  return {};
}

std::vector<double> CpuMonitor::parseLoadAverage(const std::string& text) {
  std::vector<double> values;
  for (const auto& field : sysfs::split(text, " \t\n")) {
    // /proc/loadavg is "1m 5m 15m running/total pid". Taking only the first
    // three tokens avoids mistaking the running/total pair for a load value.
    if (values.size() == 3) {
      break;
    }
    if (auto parsed = sysfs::parseDouble(field)) {
      values.push_back(*parsed);
    }
  }
  return values;
}

CpuSnapshot CpuMonitor::sample() {
  CpuSnapshot snapshot;
  snapshot.coreCount = cpuCoreCount();
  snapshot.modelName = cpuModelName();
  snapshot.frequencyMhz = averageFrequencyMhz();
  snapshot.maxFrequencyMhz = maxFrequencyMhz();

  if (const auto loadavg = sysfs::readFirstLine("/proc/loadavg")) {
    const auto values = parseLoadAverage(*loadavg);
    if (values.size() >= 3) {
      snapshot.loadAverage1 = values[0];
      snapshot.loadAverage5 = values[1];
      snapshot.loadAverage15 = values[2];
    }
  }

  const auto stat = sysfs::readFile("/proc/stat");
  if (!stat) {
    last_ = snapshot;
    return snapshot;
  }

  CpuTimes total;
  std::vector<CpuTimes> cores;
  cores.reserve(snapshot.coreCount > 0 ? static_cast<std::size_t>(snapshot.coreCount) : 8);
  for (const auto& line : sysfs::split(*stat, "\n")) {
    if (line.rfind("cpu ", 0) == 0) {
      total = parseStatLine(line);
    } else if (line.rfind("cpu", 0) == 0 && line.size() > 3 &&
               std::isdigit(static_cast<unsigned char>(line[3]))) {
      cores.push_back(parseStatLine(line));
    }
  }

  if (primed_) {
    const double totalDelta = static_cast<double>(total.total() - previousTotal_.total());
    const double idleDelta = static_cast<double>(total.idleAll() - previousTotal_.idleAll());
    const double busyDelta = totalDelta - idleDelta;

    snapshot.usagePercent = percentage(busyDelta, totalDelta);
    snapshot.userPercent =
        percentage(static_cast<double>((total.user + total.nice) - (previousTotal_.user + previousTotal_.nice)),
                   totalDelta);
    snapshot.systemPercent = percentage(static_cast<double>(total.system - previousTotal_.system), totalDelta);
    snapshot.idlePercent = percentage(idleDelta, totalDelta);
    snapshot.iowaitPercent =
        percentage(static_cast<double>(total.iowait - previousTotal_.iowait), totalDelta);

    if (!previousCores_.empty() && previousCores_.size() == cores.size()) {
      snapshot.cores.reserve(cores.size());
      for (std::size_t i = 0; i < cores.size(); ++i) {
        const double coreTotal =
            static_cast<double>(cores[i].total() - previousCores_[i].total());
        const double coreIdle =
            static_cast<double>(cores[i].idleAll() - previousCores_[i].idleAll());
        CpuCore core;
        core.index = static_cast<int>(i);
        core.usagePercent = percentage(coreTotal - coreIdle, coreTotal);
        snapshot.cores.push_back(core);
      }
    }
  }

  previousTotal_ = total;
  previousCores_ = cores;
  primed_ = true;

  if (snapshot.cores.empty()) {
    for (int i = 0; i < snapshot.coreCount; ++i) {
      CpuCore core;
      core.index = i;
      snapshot.cores.push_back(core);
    }
  }

  last_ = snapshot;
  return snapshot;
}

}  // namespace visionai
