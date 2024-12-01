#include "NVMLLibrary.h"
#include "GPUVendor.h"

#include <nvml.h>

#include <cbang/Exception.h>
#include <cbang/Catch.h>

#include <cstring>

using namespace std;
using namespace cb;


#undef CBANG_EXCEPTION
#define CBANG_EXCEPTION DynamicLibraryException

#ifdef _WIN32
static const char *nvmlLib = "nvml.dll";
#define STDCALL __stdcall

#elif __APPLE__
static const char *nvmlLib = "/Library/Frameworks/CUDA.framework/CUDA";
#define STDCALL

#else
// To link against the NVML library add the -lnvidia-ml flag to your linker command.
static const char *nvmlLib = "libnvidia-ml.so";
#define STDCALL
#endif

#ifdef _WIN32
#define NVML_API __stdcall
#else
#define NVML_API
#endif

#undef DYNAMIC_CALL
#define DYNAMIC_CALL(name, args) {                                 \
    name##_t name = (name##_t)getSymbol(#name);                    \
    if ((err = (nvmlReturn_t)name args)) THROW(#name "() returned " << (int)err);     \
  }

namespace {
  typedef nvmlReturn_t (NVML_API *nvmlInit_v2_t)();
  typedef nvmlReturn_t (NVML_API *nvmlShutdown_t)();

  typedef nvmlReturn_t (NVML_API *nvmlSystemGetCudaDriverVersion_v2_t)(int*);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetCount_v2_t)(unsigned int*);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetHandleByIndex_v2_t)(unsigned int, nvmlDevice_t*);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetHandleByUUID_t)(const char*, nvmlDevice_t*);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetUUID_t)(nvmlDevice_t, char*, unsigned int);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetCudaComputeCapability_t)(nvmlDevice_t, int*, int*);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetPciInfo_v3_t)(nvmlDevice_t, nvmlPciInfo_t*);

  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetCurrPcieLinkWidth_t)(nvmlDevice_t, unsigned int*);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetMaxPcieLinkWidth_t)(nvmlDevice_t, unsigned int*);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetCurrPcieLinkGeneration_t)(nvmlDevice_t, unsigned int*);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetMaxPcieLinkGeneration_t)(nvmlDevice_t, unsigned int*);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetGpuMaxPcieLinkGeneration_t)(nvmlDevice_t, unsigned int*);

  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetClock_t)(nvmlDevice_t, nvmlClockType_t, nvmlClockId_t, unsigned int*);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetMaxClockInfo_t)(nvmlDevice_t, nvmlClockType_t, unsigned int*);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetTemperature_t)(nvmlDevice_t, nvmlTemperatureSensors_t, unsigned int*);
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetPerformanceState_t)(nvmlDevice_t, nvmlPstates_t*);
}

NVMLLibrary::NVMLLibrary(Inaccessible) : DynamicLibrary(nvmlLib) {
  nvmlReturn_t err;
  int version;

  DYNAMIC_CALL(nvmlInit_v2, ());
  DYNAMIC_CALL(nvmlSystemGetCudaDriverVersion_v2, (&version));

  VersionU16 driverVersion(version / 1000, (version % 1000) / 10);

  unsigned int count = 0;
  DYNAMIC_CALL(nvmlDeviceGetCount_v2, (&count));

  for (unsigned int i = 0; i < count; i++) {
    ComputeDevice cd;

    cd.platform = "NVML";

    // Set indices
    cd.platformIndex = 0; // Only one platform for CUDA
    cd.deviceIndex = i;
    cd.gpu = true; // All CUDA devices are GPUs
    cd.vendorID = GPUVendor::VENDOR_NVIDIA; // Only vendor for CUDA
    cd.driverVersion = driverVersion;
    cd.pciFunction = 0; // NVidia GPUs are always function 0

    try {
      nvmlDevice_t device;
      DYNAMIC_CALL(nvmlDeviceGetHandleByIndex_v2, (i, &device));

      int major = 0, minor = 0;
      DYNAMIC_CALL(nvmlDeviceGetCudaComputeCapability, (device, &major, &minor));
      cd.computeVersion = VersionU16(major, minor);

      nvmlPciInfo_t pciInfo;
      DYNAMIC_CALL(nvmlDeviceGetPciInfo_v3, (device, &pciInfo));
      cd.pciBus  = pciInfo.bus;
      cd.pciSlot = pciInfo.device;

//      CUuuid uuid = {{0,}};
//      DYNAMIC_CALL(cuDeviceGetUuid, (&uuid, device));
//      cd.uuid = UUID(uuid.bytes);

      // HACK: The UUID returned is a string prefixed with GPU so placing this in the name
      //       property though this is an off-label usage, intended to be used as a lookup for nvmlDevice_t
      const unsigned len = 100;
      char name[len];
      DYNAMIC_CALL(nvmlDeviceGetUUID, (device, name, len));
      cd.name = string(name, strnlen(name, len));

      devices.push_back(cd);
    } CATCH_ERROR;
  }
}

NVMLLibrary::~NVMLLibrary() {
  nvmlReturn_t err;
  try {
    DYNAMIC_CALL(nvmlShutdown, ());
  } CATCH_ERROR;
}

const ComputeDevice &NVMLLibrary::getDevice(unsigned i) const {
  if (getDeviceCount() <= i) THROW("Invalid NVML device index " << i);
  return devices.at(i);
}

bool NVMLLibrary::tryGetMeasurements(const char* uuid, GPUMeasurement &measurements) {
  nvmlReturn_t err;
  nvmlDevice_t device;
  unsigned int value = 0;
  nvmlPstates_t pstate;

  try {
    DYNAMIC_CALL(nvmlDeviceGetHandleByUUID, (uuid, &device));

    DYNAMIC_CALL(nvmlDeviceGetClock, (device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &value));
    measurements.gpuFreq_MHz = (uint16_t)value;
    DYNAMIC_CALL(nvmlDeviceGetMaxClockInfo, (device, NVML_CLOCK_GRAPHICS, &value));
    measurements.gpuFreqLimit_MHz = (uint16_t)value;
    DYNAMIC_CALL(nvmlDeviceGetClock, (device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT, &value));
    measurements.memFreq_MHz = (uint16_t)value;
    DYNAMIC_CALL(nvmlDeviceGetMaxClockInfo, (device, NVML_CLOCK_MEM, &value));
    measurements.memFreqLimit_MHz = (uint16_t)value;
    DYNAMIC_CALL(nvmlDeviceGetTemperature, (device, NVML_TEMPERATURE_GPU, &value));
    measurements.gpuTemp_C = (uint8_t)value;
    DYNAMIC_CALL(nvmlDeviceGetPerformanceState, (device, &pstate));
    measurements.pstate = (uint8_t)pstate;
    DYNAMIC_CALL(nvmlDeviceGetCurrPcieLinkGeneration, (device, &value));
    measurements.currPCIeLinkGen = (uint8_t)value;
    DYNAMIC_CALL(nvmlDeviceGetMaxPcieLinkGeneration, (device, &value));
    measurements.maxPCIeLinkGen = (uint8_t)value;
    DYNAMIC_CALL(nvmlDeviceGetGpuMaxPcieLinkGeneration, (device, &value));
    measurements.maxPCIeLinkGenDevice = (uint8_t)value;
    DYNAMIC_CALL(nvmlDeviceGetCurrPcieLinkWidth, (device, &value));
    measurements.currPCIeLinkWidth = (uint8_t)value;
    DYNAMIC_CALL(nvmlDeviceGetMaxPcieLinkWidth, (device, &value));
    measurements.maxPCIeLinkWidth = (uint8_t)value;

    // We got everything 
    return true;
  } CATCH_ERROR;

  // Something failed, struct contents are indeterminant
  return false;
}
