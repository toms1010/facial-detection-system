/* Stable C ABI for the hardware layer.
 *
 * The pybind11 module is the primary interface, but a pybind11 build requires
 * matching Python headers. This C ABI lets the same C++ code be loaded through
 * ctypes with no build-time Python dependency, which is what the pure-fallback
 * path in visionai/hardware/native_loader.py uses.
 *
 * Payloads are JSON documents; the caller owns the returned string and must
 * release it with visionai_free_string().
 */
#ifndef VISIONAI_C_API_H
#define VISIONAI_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* Returns a malloc'd JSON snapshot. Release with visionai_free_string. */
char* visionai_snapshot_json(const char* disk_path, const char* network_interface);

/* Returns a malloc'd JSON document for a single subsystem:
 * "cpu", "memory", "temperature", "disk", "network", "gpus" or "system". */
char* visionai_subsystem_json(const char* subsystem, const char* argument);

/* Library version string. Release with visionai_free_string. */
char* visionai_version_json(void);

/* Comma separated capability list. Release with visionai_free_string. */
char* visionai_capabilities_json(void);

void visionai_free_string(char* text);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* VISIONAI_C_API_H */
