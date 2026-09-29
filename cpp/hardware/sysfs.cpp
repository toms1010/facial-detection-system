#include "sysfs.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <system_error>

namespace fs = std::filesystem;

namespace visionai::sysfs {

std::optional<std::string> readFile(const std::string& path) {
  std::ifstream stream(path);
  if (!stream.is_open()) {
    return std::nullopt;
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

std::optional<std::string> readFirstLine(const std::string& path) {
  const auto content = readFile(path);
  if (!content) {
    return std::nullopt;
  }
  const auto newline = content->find('\n');
  const std::string line = newline == std::string::npos ? *content : content->substr(0, newline);
  const std::string trimmed = trim(line);
  if (trimmed.empty()) {
    return std::nullopt;
  }
  return trimmed;
}

std::optional<double> parseDouble(std::string_view text) {
  const std::string cleaned = trim(text);
  if (cleaned.empty()) {
    return std::nullopt;
  }
  char* end = nullptr;
  const double value = std::strtod(cleaned.c_str(), &end);
  if (end == cleaned.c_str()) {
    return std::nullopt;
  }
  return value;
}

std::optional<std::uint64_t> parseUint(std::string_view text) {
  const std::string cleaned = trim(text);
  if (cleaned.empty()) {
    return std::nullopt;
  }
  std::uint64_t value = 0;
  const auto* first = cleaned.data();
  const auto* last = cleaned.data() + cleaned.size();
  const auto result = std::from_chars(first, last, value);
  if (result.ec != std::errc{}) {
    return std::nullopt;
  }
  return value;
}

std::optional<double> readTemperatureCelsius(const std::string& path) {
  const auto raw = readFirstLine(path);
  if (!raw) {
    return std::nullopt;
  }
  // hwmon temp*_input and thermal_zone temp are millidegrees by definition.
  // A handful of drivers write "50.0" instead, so treat a decimal point as
  // an explicit degrees value and otherwise divide by 1000.
  if (raw->find('.') != std::string::npos) {
    const auto celsius = parseDouble(*raw);
    if (!celsius) {
      return std::nullopt;
    }
    if (*celsius < kMinPlausibleCelsius || *celsius > kMaxPlausibleCelsius) {
      return std::nullopt;
    }
    return celsius;
  }
  const auto milli = parseUint(*raw);
  if (!milli) {
    return std::nullopt;
  }
  const double celsius = static_cast<double>(*milli) / 1000.0;
  if (celsius < kMinPlausibleCelsius || celsius > kMaxPlausibleCelsius) {
    return std::nullopt;
  }
  return celsius;
}

std::vector<std::string> split(std::string_view text, std::string_view delimiters) {
  std::vector<std::string> parts;
  std::string current;
  for (const char c : text) {
    if (delimiters.find(c) != std::string_view::npos) {
      if (!current.empty()) {
        parts.push_back(current);
        current.clear();
      }
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) {
    parts.push_back(current);
  }
  return parts;
}

std::string trim(std::string_view text) {
  const auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
  std::size_t begin = 0;
  while (begin < text.size() && isSpace(static_cast<unsigned char>(text[begin]))) {
    ++begin;
  }
  std::size_t end = text.size();
  while (end > begin && isSpace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }
  return std::string(text.substr(begin, end - begin));
}

std::vector<std::string> listDirectories(const std::string& path) {
  std::vector<std::string> names;
  std::error_code ec;
  if (!fs::is_directory(path, ec)) {
    return names;
  }
  for (fs::directory_iterator it(path, ec), end; it != end && !ec; it.increment(ec)) {
    if (it->is_directory(ec)) {
      names.push_back(it->path().filename().string());
    }
  }
  std::sort(names.begin(), names.end());
  return names;
}

bool pathExists(const std::string& path) {
  std::error_code ec;
  return fs::exists(path, ec);
}

std::optional<std::string> readLinkTarget(const std::string& path) {
  std::error_code ec;
  const auto target = fs::read_symlink(path, ec);
  if (ec) {
    return std::nullopt;
  }
  const std::string text = target.string();
  const auto slash = text.find_last_of('/');
  return slash == std::string::npos ? text : text.substr(slash + 1);
}

}  // namespace visionai::sysfs
