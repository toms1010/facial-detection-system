// Small helpers for reading Linux procfs/sysfs without throwing.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace visionai::sysfs {

// Reads a whole file. Returns nullopt when it cannot be read.
std::optional<std::string> readFile(const std::string& path);

// Reads the first line of a file, trimmed.
std::optional<std::string> readFirstLine(const std::string& path);

// Parses a leading float from a string.
std::optional<double> parseDouble(std::string_view text);

// Parses a leading unsigned 64-bit integer from a string.
std::optional<std::uint64_t> parseUint(std::string_view text);

// Reads a Linux temperature file (hwmon temp*_input or thermal_zone temp).
// Those files are in millidegrees unless the driver writes a decimal value,
// and both encodings appear in the wild, so the scale is detected from the text.
std::optional<double> readTemperatureCelsius(const std::string& path);

// Plausible range for a real silicon temperature, used to reject bad scales.
inline constexpr double kMinPlausibleCelsius = -40.0;
inline constexpr double kMaxPlausibleCelsius = 150.0;

// Splits on any of the given delimiters.
std::vector<std::string> split(std::string_view text, std::string_view delimiters);

std::string trim(std::string_view text);

// Lists immediate subdirectory names of a directory.
std::vector<std::string> listDirectories(const std::string& path);

bool pathExists(const std::string& path);

// Resolves a symlink to its final component, e.g. the "i915" in
// /sys/class/drm/card0/device/driver. Returns nullopt when not a symlink.
std::optional<std::string> readLinkTarget(const std::string& path);

}  // namespace visionai::sysfs
