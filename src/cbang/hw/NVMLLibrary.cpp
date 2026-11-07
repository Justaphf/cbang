/******************************************************************************\

          This file is part of the C! library.  A.K.A the cbang library.

                Copyright (c) 2021-2026, Cauldron Development  Oy
                Copyright (c) 2003-2021, Cauldron Development LLC
                               All rights reserved.

         The C! library is free software: you can redistribute it and/or
        modify it under the terms of the GNU Lesser General Public License
       as published by the Free Software Foundation, either version 2.1 of
               the License, or (at your option) any later version.

        The C! library is distributed in the hope that it will be useful,
          but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
                 Lesser General Public License for more details.

         You should have received a copy of the GNU Lesser General Public
                 License along with the C! library.  If not, see
                         <http://www.gnu.org/licenses/>.

        In addition, BSD licensing may be granted on a case by case basis
        by written permission from at least one of the copyright holders.
           You may request written permission by emailing the authors.

                  For information regarding this software email:
                                 Joseph Coffland
                          joseph@cauldrondevelopment.com

\******************************************************************************/

#include "NVMLLibrary.h"
#include "GPUVendor.h"

#include <cbang/Exception.h>
#include <cbang/Catch.h>
#include <cbang/String.h>

#ifdef _WIN32
#include <cbang/os/SystemUtilities.h>
#endif

#include <cstring>

using namespace std;
using namespace cb;


#undef CBANG_EXCEPTION
#define CBANG_EXCEPTION DynamicLibraryException

// Function pointer for name, or 0
#define DYNAMIC_FIND(name)                                          \
  reinterpret_cast<name##_t>(findSymbol(#name))

#define DYNAMIC_CALL(name, args) {                                  \
    auto name = DYNAMIC_FIND(name);                                 \
    if (!name) THROW("NVML function " #name "() not found");        \
    if ((err = name args)) THROW(#name "() returned " << err);      \
  }

// Evaluates to false if the driver or device does not support the call
#define DYNAMIC_READ(name, args)                                    \
  supported(orNotFound(DYNAMIC_FIND(name)) args, #name)


namespace {
  // The oldest driver supported is R470, Kepler's last branch.  Fallbacks:
  //   nvmlSystemGetCudaDriverVersion where _v2 finds no CUDA driver library;
  //   fans probed by index before nvmlDeviceGetNumFans (R510), and the first
  //   fan via nvmlDeviceGetFanSpeed where _v2 reads none;
  //   nvmlDeviceGetTemperature before nvmlDeviceGetTemperatureV (R565).
  //   Other missing functions, e.g. nvmlDeviceGetGpuMaxPcieLinkGeneration
  //   (R525), leave their field unset.
  using nvmlDevice_t = struct nvmlDevice_st *;

  // Layout must match nvml.h
  struct nvmlPciInfo_t {
    char busIdLegacy[16];
    unsigned domain;
    unsigned bus;
    unsigned device;
    unsigned pciDeviceId;
    unsigned pciSubSystemId;
    char busId[32];
  };

  // Layout must match nvml.h
  struct nvmlUtilization_t {
    unsigned gpu;
    unsigned memory;
  };

  // Layout must match nvml.h
  struct nvmlTemperature_t {
    unsigned version;
    int sensorType;
    int temperature;
  };

  using nvmlInit_v2_t = int (*)();
  using nvmlShutdown_t = int (*)();
  using nvmlSystemGetCudaDriverVersion_v2_t = int (*)(int *);
  using nvmlSystemGetCudaDriverVersion_t = int (*)(int *);
  using nvmlSystemGetDriverVersion_t = int (*)(char *, unsigned);
  using nvmlDeviceGetCount_v2_t = int (*)(unsigned *);
  using nvmlDeviceGetHandleByIndex_v2_t = int (*)(unsigned, nvmlDevice_t *);
  using nvmlDeviceGetHandleByUUID_t = int (*)(const char *, nvmlDevice_t *);
  using nvmlDeviceGetUUID_t = int (*)(nvmlDevice_t, char *, unsigned);
  using nvmlDeviceGetName_t = int (*)(nvmlDevice_t, char *, unsigned);
  using nvmlDeviceGetCudaComputeCapability_t =
    int (*)(nvmlDevice_t, int *, int *);
  using nvmlDeviceGetPciInfo_v3_t = int (*)(nvmlDevice_t, nvmlPciInfo_t *);
  using nvmlDeviceGetClock_t = int (*)(nvmlDevice_t, int, int, unsigned *);
  using nvmlDeviceGetMaxClockInfo_t = int (*)(nvmlDevice_t, int, unsigned *);
  using nvmlDeviceGetPowerUsage_t = int (*)(nvmlDevice_t, unsigned *);
  using nvmlDeviceGetEnforcedPowerLimit_t = int (*)(nvmlDevice_t, unsigned *);
  using nvmlDeviceGetTemperature_t = int (*)(nvmlDevice_t, int, unsigned *);
  using nvmlDeviceGetTemperatureV_t =
    int (*)(nvmlDevice_t, nvmlTemperature_t *);
  using nvmlDeviceGetUtilizationRates_t =
    int (*)(nvmlDevice_t, nvmlUtilization_t *);
  using nvmlDeviceGetPerformanceState_t = int (*)(nvmlDevice_t, int *);
  using nvmlDeviceGetCurrPcieLinkGeneration_t =
    int (*)(nvmlDevice_t, unsigned *);
  using nvmlDeviceGetMaxPcieLinkGeneration_t =
    int (*)(nvmlDevice_t, unsigned *);
  using nvmlDeviceGetGpuMaxPcieLinkGeneration_t =
    int (*)(nvmlDevice_t, unsigned *);
  using nvmlDeviceGetCurrPcieLinkWidth_t = int (*)(nvmlDevice_t, unsigned *);
  using nvmlDeviceGetMaxPcieLinkWidth_t = int (*)(nvmlDevice_t, unsigned *);
  using nvmlDeviceGetNumFans_t = int (*)(nvmlDevice_t, unsigned *);
  using nvmlDeviceGetFanSpeed_v2_t =
    int (*)(nvmlDevice_t, unsigned, unsigned *);
  using nvmlDeviceGetFanSpeed_t = int (*)(nvmlDevice_t, unsigned *);

  const int NVML_ERROR_INVALID_ARGUMENT = 2;
  const int NVML_ERROR_NOT_SUPPORTED = 3;
  const int NVML_ERROR_LIBRARY_NOT_FOUND = 12;
  const int NVML_ERROR_FUNCTION_NOT_FOUND = 13;
  const int NVML_ERROR_ARGUMENT_VERSION_MISMATCH = 25;
  const int NVML_CLOCK_GRAPHICS = 0;
  const int NVML_CLOCK_MEM = 2;
  const int NVML_CLOCK_ID_CURRENT = 0;
  const int NVML_TEMPERATURE_GPU = 0;
  const int NVML_PSTATE_UNKNOWN = 32;
  const unsigned NVML_DEVICE_UUID_V2_BUFFER_SIZE = 96;
  const unsigned NVML_DEVICE_NAME_V2_BUFFER_SIZE = 96;
  const unsigned NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE = 80;
  const unsigned nvmlTemperature_v1 =
    static_cast<unsigned>(sizeof(nvmlTemperature_t)) | 1U << 24;


  bool noCUDA(int err) {
    return err == NVML_ERROR_LIBRARY_NOT_FOUND ||
      err == NVML_ERROR_FUNCTION_NOT_FOUND;
  }


  bool supported(int err, const char *name) {
    if (err == NVML_ERROR_NOT_SUPPORTED ||
        err == NVML_ERROR_FUNCTION_NOT_FOUND) return false;
    if (err) THROW(name << "() returned " << err);
    return true;
  }


  // Stands in for a function the driver does not export
  template <typename... Args>
  int notFound(Args...) {return NVML_ERROR_FUNCTION_NOT_FOUND;}


  template <typename... Args>
  auto orNotFound(int (*func)(Args...)) -> int (*)(Args...) {
    if (func) return func;
    return notFound<Args...>;
  }


  vector<string> getLibraryPaths() {
#ifdef _WIN32
    vector<string> paths = {"nvml.dll"};

    // Pre-DCH drivers install NVML outside of the DLL search path
    const char *dir = SystemUtilities::getenv("ProgramW6432");
    if (!dir) dir = SystemUtilities::getenv("ProgramFiles");
    if (dir)
      paths.push_back(string(dir) + "\\NVIDIA Corporation\\NVSMI\\nvml.dll");

    return paths;

#elif __APPLE__
    return {"libnvidia-ml.dylib"}; // There is no NVML for macOS

#else
    return {"libnvidia-ml.so.1", "libnvidia-ml.so"};
#endif
  }
}


NVMLLibrary::NVMLLibrary(Inaccessible) : DynamicLibrary(getLibraryPaths()) {
  int err;
  int version = 0;
  unsigned count = 0;
  VersionU16 driverVersion;

  DYNAMIC_CALL(nvmlInit_v2, ());

  try {
    // _v2 fails without the CUDA driver library.  The older call then reports
    // a supported version, else the NVIDIA driver version keeps the devices
    // valid.
    auto getVersion = DYNAMIC_FIND(nvmlSystemGetCudaDriverVersion_v2);
    err = orNotFound(getVersion)(&version);

    if (noCUDA(err)) {
      getVersion = DYNAMIC_FIND(nvmlSystemGetCudaDriverVersion);
      err = orNotFound(getVersion)(&version);
    }

    if (noCUDA(err)) {
      const unsigned versionLen = NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE;
      char buf[versionLen];
      DYNAMIC_CALL(nvmlSystemGetDriverVersion, (buf, versionLen));
      driverVersion = VersionU16(string(buf, strnlen(buf, versionLen)));

    } else if (err)
      THROW("nvmlSystemGetCudaDriverVersion() returned " << err);
    else driverVersion = VersionU16(version / 1000, (version % 1000) / 10);

    DYNAMIC_CALL(nvmlDeviceGetCount_v2, (&count));

  } catch (...) {
    // The destructor will not run
    try {DYNAMIC_CALL(nvmlShutdown, ());} CATCH_ERROR;
    throw;
  }

  for (unsigned i = 0; i < count; i++) {
    ComputeDevice cd;

    cd.platform = "NVML";

    // Set indices
    cd.platformIndex = 0; // Only one platform for NVML
    cd.deviceIndex = i;
    cd.gpu = true; // All NVML devices are GPUs

    try {
      nvmlDevice_t device = 0;

      DYNAMIC_CALL(nvmlDeviceGetHandleByIndex_v2, (i, &device));

      int major = 0;
      int minor = 0;
      DYNAMIC_CALL(nvmlDeviceGetCudaComputeCapability,
                   (device, &major, &minor));

      cd.driverVersion = driverVersion;
      cd.computeVersion = VersionU16(major, minor);
      cd.vendorID = GPUVendor::VENDOR_NVIDIA; // Only vendor for NVML

      nvmlPciInfo_t pciInfo;
      DYNAMIC_CALL(nvmlDeviceGetPciInfo_v3, (device, &pciInfo));
      cd.pciDomain = pciInfo.domain;
      cd.pciBus = pciInfo.bus;
      cd.pciSlot = pciInfo.device;
      cd.pciFunction = 0; // NVidia GPUs are always function 0

      const unsigned uuidLen = NVML_DEVICE_UUID_V2_BUFFER_SIZE;
      char uuid[uuidLen];
      DYNAMIC_CALL(nvmlDeviceGetUUID, (device, uuid, uuidLen));

      // Match the CUDA and OpenCL form, i.e. without the "GPU-" prefix
      cd.uuid = String::toLower(string(uuid, strnlen(uuid, uuidLen)));
      if (String::startsWith(cd.uuid, "gpu-")) cd.uuid = cd.uuid.substr(4);

      const unsigned nameLen = NVML_DEVICE_NAME_V2_BUFFER_SIZE;
      char name[nameLen];
      DYNAMIC_CALL(nvmlDeviceGetName, (device, name, nameLen));
      cd.name = string(name, strnlen(name, nameLen));

      devices.push_back(cd);
    } CATCH_ERROR;
  }
}


NVMLLibrary::~NVMLLibrary() {
  int err;

  try {
    DYNAMIC_CALL(nvmlShutdown, ());
  } CATCH_ERROR;
}


const ComputeDevice &NVMLLibrary::getDevice(unsigned i) const {
  if (getDeviceCount() <= i) THROW("Invalid NVML device index " << i);
  return devices.at(i);
}


GPUMeasurement NVMLLibrary::getMeasurement(const string &uuid) {
  int err;
  nvmlDevice_t device = 0;
  DYNAMIC_CALL(nvmlDeviceGetHandleByUUID, (("GPU-" + uuid).c_str(), &device));

  GPUMeasurement m;
  unsigned value = 0;

  if (DYNAMIC_READ(nvmlDeviceGetClock, (device, NVML_CLOCK_GRAPHICS,
                                        NVML_CLOCK_ID_CURRENT, &value)))
    m.gpuClock = value;
  if (DYNAMIC_READ(nvmlDeviceGetMaxClockInfo,
                   (device, NVML_CLOCK_GRAPHICS, &value)))
    m.gpuClockMax = value;
  if (DYNAMIC_READ(nvmlDeviceGetClock, (device, NVML_CLOCK_MEM,
                                        NVML_CLOCK_ID_CURRENT, &value)))
    m.memClock = value;
  if (DYNAMIC_READ(nvmlDeviceGetMaxClockInfo,
                   (device, NVML_CLOCK_MEM, &value)))
    m.memClockMax = value;

  // NVML reports milliwatts
  if (DYNAMIC_READ(nvmlDeviceGetPowerUsage, (device, &value)))
    m.power = value / 1000.0;
  if (DYNAMIC_READ(nvmlDeviceGetEnforcedPowerLimit, (device, &value)))
    m.powerLimit = value / 1000.0;

  nvmlTemperature_t temp = {nvmlTemperature_v1, NVML_TEMPERATURE_GPU, 0};
  err = orNotFound(DYNAMIC_FIND(nvmlDeviceGetTemperatureV))(device, &temp);

  if (err == NVML_ERROR_FUNCTION_NOT_FOUND ||
      err == NVML_ERROR_ARGUMENT_VERSION_MISMATCH) {
    if (DYNAMIC_READ(nvmlDeviceGetTemperature,
                     (device, NVML_TEMPERATURE_GPU, &value)))
      m.temperature = value;

  } else if (supported(err, "nvmlDeviceGetTemperatureV") &&
             0 < temp.temperature) m.temperature = temp.temperature;

  nvmlUtilization_t utilization;
  if (DYNAMIC_READ(nvmlDeviceGetUtilizationRates, (device, &utilization))) {
    m.utilization = utilization.gpu;
    m.memUtilization = utilization.memory;
  }

  int pstate = NVML_PSTATE_UNKNOWN;
  if (DYNAMIC_READ(nvmlDeviceGetPerformanceState, (device, &pstate)) &&
      pstate != NVML_PSTATE_UNKNOWN) m.pstate = pstate;

  if (DYNAMIC_READ(nvmlDeviceGetCurrPcieLinkGeneration, (device, &value)))
    m.pcieGen = value;
  if (DYNAMIC_READ(nvmlDeviceGetMaxPcieLinkGeneration, (device, &value)))
    m.pcieGenMax = value;
  if (DYNAMIC_READ(nvmlDeviceGetGpuMaxPcieLinkGeneration, (device, &value)))
    m.pcieGenDeviceMax = value;
  if (DYNAMIC_READ(nvmlDeviceGetCurrPcieLinkWidth, (device, &value)))
    m.pcieWidth = value;
  if (DYNAMIC_READ(nvmlDeviceGetMaxPcieLinkWidth, (device, &value)))
    m.pcieWidthMax = value;

  // 0% is a real reading, so unreadable fans are left out.  Stopping at the
  // first one keeps the fan indices.  Without the fan count up to 8 fans are
  // probed until an index is invalid.  The legacy call reads the first fan
  // where _v2 or the fan count is not supported.
  unsigned numFans = 0;
  bool probe = !DYNAMIC_FIND(nvmlDeviceGetNumFans);
  bool counted = DYNAMIC_READ(nvmlDeviceGetNumFans, (device, &numFans));
  if (probe) numFans = 8;

  auto getFanSpeed = orNotFound(DYNAMIC_FIND(nvmlDeviceGetFanSpeed_v2));
  for (unsigned i = 0; (counted || probe) && i < numFans; i++) {
    err = getFanSpeed(device, i, &value);
    if (probe && err == NVML_ERROR_INVALID_ARGUMENT) break;
    if (!supported(err, "nvmlDeviceGetFanSpeed_v2")) break;
    m.fans.push_back(value);
  }

  if (m.fans.empty() && (!counted || numFans) &&
      DYNAMIC_READ(nvmlDeviceGetFanSpeed, (device, &value)))
    m.fans.push_back(value);

  return m;
}


#undef DYNAMIC_FIND
#undef DYNAMIC_CALL
#undef DYNAMIC_READ
