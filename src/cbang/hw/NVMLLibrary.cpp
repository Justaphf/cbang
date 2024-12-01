#include "NVMLLibrary.h"
#include "GPUVendor.h"

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
static vector<string> nvmlLib = { "libnvidia-ml.so.1", "libnvidia-ml.so" };
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

/**
 * Buffer size guaranteed to be large enough for pci bus id
 */
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE      32 //!< Buffer size for PCI bus ID.

/**
 * Buffer size guaranteed to be large enough for pci bus id for \p busIdLegacy
 */
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE   16 //!< Buffer size for legacy PCI bus ID.

namespace {
  /*
   * NOTE: These typedefs were pulled directly from nvml.h as installed on Windows for the CUDA toolkit v13.1
   *       Done this way to avoid having to pull in the header file when building, similar to CUDALibrary/OpenCLLibrary/HIPLibrary
   */
  typedef struct nvmlDevice_st* nvmlDevice_t;

  /**
   * Return values for NVML API calls.
   */
  typedef enum nvmlReturn_enum
  {
    // cppcheck-suppress *
    NVML_SUCCESS = 0,                          //!< The operation was successful
    NVML_ERROR_UNINITIALIZED = 1,              //!< NVML was not first initialized with nvmlInit()
    NVML_ERROR_INVALID_ARGUMENT = 2,           //!< A supplied argument is invalid
    NVML_ERROR_NOT_SUPPORTED = 3,              //!< The requested operation is not available on target device
    NVML_ERROR_NO_PERMISSION = 4,              //!< The current user does not have permission for operation
    NVML_ERROR_ALREADY_INITIALIZED = 5,        //!< Deprecated: Multiple initializations are now allowed through ref counting
    NVML_ERROR_NOT_FOUND = 6,                  //!< A query to find an object was unsuccessful
    NVML_ERROR_INSUFFICIENT_SIZE = 7,          //!< An input argument is not large enough
    NVML_ERROR_INSUFFICIENT_POWER = 8,         //!< A device's external power cables are not properly attached
    NVML_ERROR_DRIVER_NOT_LOADED = 9,          //!< NVIDIA driver is not loaded
    NVML_ERROR_TIMEOUT = 10,                   //!< User provided timeout passed
    NVML_ERROR_IRQ_ISSUE = 11,                 //!< NVIDIA Kernel detected an interrupt issue with a GPU
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,         //!< NVML Shared Library couldn't be found or loaded
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,        //!< Local version of NVML doesn't implement this function
    NVML_ERROR_CORRUPTED_INFOROM = 14,         //!< infoROM is corrupted
    NVML_ERROR_GPU_IS_LOST = 15,               //!< The GPU has fallen off the bus or has otherwise become inaccessible
    NVML_ERROR_RESET_REQUIRED = 16,            //!< The GPU requires a reset before it can be used again
    NVML_ERROR_OPERATING_SYSTEM = 17,          //!< The GPU control device has been blocked by the operating system/cgroups
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,   //!< RM detects a driver/library version mismatch
    NVML_ERROR_IN_USE = 19,                    //!< An operation cannot be performed because the GPU is currently in use
    NVML_ERROR_MEMORY = 20,                    //!< Insufficient memory
    NVML_ERROR_NO_DATA = 21,                   //!< No data
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,    //!< The requested vgpu operation is not available on target device, becasue ECC is enabled
    NVML_ERROR_INSUFFICIENT_RESOURCES = 23,    //!< Ran out of critical resources, other than memory
    NVML_ERROR_FREQ_NOT_SUPPORTED = 24,        //!< Ran out of critical resources, other than memory
    NVML_ERROR_ARGUMENT_VERSION_MISMATCH = 25, //!< The provided version is invalid/unsupported
    NVML_ERROR_DEPRECATED  = 26,               //!< The requested functionality has been deprecated
    NVML_ERROR_NOT_READY = 27,                 //!< The system is not ready for the request
    NVML_ERROR_GPU_NOT_FOUND = 28,             //!< No GPUs were found
    NVML_ERROR_INVALID_STATE = 29,             //!< Resource not in correct state to perform requested operation
    NVML_ERROR_RESET_TYPE_NOT_SUPPORTED = 30,       //!< Reset not supported for given device/parameters
    NVML_ERROR_UNKNOWN = 999                   //!< An internal driver error occurred
  } nvmlReturn_t;

  /**
   * PCI information about a GPU device.
   */
  typedef struct nvmlPciInfo_st
  {
    char busIdLegacy[NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE]; //!< The legacy tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
    unsigned int domain;             //!< The PCI domain on which the device's bus resides, 0 to 0xffffffff
    unsigned int bus;                //!< The bus on which the device resides, 0 to 0xff
    unsigned int device;             //!< The device's id on the bus, 0 to 31
    unsigned int pciDeviceId;        //!< The combined 16-bit device id and 16-bit vendor id

    // Added in NVML 2.285 API
    unsigned int pciSubSystemId;     //!< The 32-bit Sub System Device ID

    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; //!< The tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
  } nvmlPciInfo_t;


  /* NVML API reference (initialization and clean-up): https://docs.nvidia.com/deploy/nvml-api/group__nvmlInitializationAndCleanup.html#group__nvmlInitializationAndCleanup */
  /* Updated: 2025-12-09
    nvmlReturn_t nvmlInit_v2 ( void )
    Returns
        NVML_SUCCESS if NVML has been properly initialized
        NVML_ERROR_DRIVER_NOT_LOADED if NVIDIA driver is not running
        NVML_ERROR_NO_PERMISSION if NVML does not have permission to talk to the driver
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Initialize NVML, but don't initialize any GPUs yet.

        Note:
        nvmlInit_v3 introduces a "flags" argument, that allows passing boolean values modifying the behaviour of nvmlInit().

        In NVML 5.319 new nvmlInit_v2 has replaced nvmlInit"_v1" (default in NVML 4.304 and older) that did initialize all GPU devices in the system.

        This allows NVML to communicate with a GPU when other GPUs in the system are unstable or in a bad state. When using this API,
        GPUs are discovered and initialized in nvmlDeviceGetHandleBy* functions instead.

        Note:
        To contrast nvmlInit_v2 with nvmlInit"_v1", NVML 4.304 nvmlInit"_v1" will fail when any detected GPU is in a bad or unstable state.

        For all products.

        This method, should be called once before invoking any other methods in the library. A reference count of the number of initializations
        is maintained. Shutdown only occurs when the reference count reaches zero.
  */
  typedef nvmlReturn_t (NVML_API *nvmlInit_v2_t)();
  /* Updated: 2025-12-09
    nvmlReturn_t nvmlShutdown ( void )
    Returns
        NVML_SUCCESS if NVML has been properly shut down
        NVML_ERROR_UNINITIALIZED if the library has not been successfully initialized
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Shut down NVML by releasing all GPU resources previously allocated with nvmlInit_v2().

        For all products.

        This method should be called after NVML work is done, once for each call to nvmlInit_v2() A reference count of the number of initializations
        is maintained. Shutdown only occurs when the reference count reaches zero. For backwards compatibility, no error is reported if
        nvmlShutdown() is called more times than nvmlInit().
  */
  typedef nvmlReturn_t (NVML_API *nvmlShutdown_t)();


  /* NVML API reference (system queries): https://docs.nvidia.com/deploy/nvml-api/group__nvmlSystemQueries.html#group__nvmlSystemQueries */
  /* Updated: 2025-12-09
    nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2 ( int* cudaDriverVersion )
    Parameters
        cudaDriverVersion
            Reference in which to return the version identifier
    Returns
        NVML_SUCCESS if cudaDriverVersion has been set
        NVML_ERROR_INVALID_ARGUMENT if cudaDriverVersion is NULL
        NVML_ERROR_LIBRARY_NOT_FOUND if libcuda.so.1 or libcuda.dll is not found
        NVML_ERROR_FUNCTION_NOT_FOUND if cuDriverGetVersion() is not found in the shared library
    Description
        Retrieves the version of the CUDA driver from the shared library.

        For all products.

        The returned CUDA driver version by calling cuDriverGetVersion()
  */
  typedef nvmlReturn_t (NVML_API *nvmlSystemGetCudaDriverVersion_v2_t)(int*);


  /* NVML API reference (device queries): https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries */
  /* Updated: 2025-12-09
    nvmlReturn_t nvmlDeviceGetCount_v2 ( unsigned int* deviceCount )
    Parameters
        deviceCount
            Reference in which to return the number of accessible devices
    Returns
        NVML_SUCCESS if deviceCount has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully initialized
        NVML_ERROR_INVALID_ARGUMENT if deviceCount is NULL
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the number of compute devices in the system. A compute device is a single GPU.

        For all products.

        Note: New nvmlDeviceGetCount_v2 (default in NVML 5.319) returns count of all devices in the system even if nvmlDeviceGetHandleByIndex_v2
        returns NVML_ERROR_NO_PERMISSION for such device. Update your code to handle this error, or use NVML 4.304 or older nvml header file. For
        backward binary compatibility reasons _v1 version of the API is still present in the shared library. Old _v1 version of nvmlDeviceGetCount
        doesn't count devices that NVML has no permission to talk to.
   */
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetCount_v2_t)(unsigned int*);
  /* Updated: 2025-12-09
    nvmlReturn_t nvmlDeviceGetHandleByIndex_v2 ( unsigned int  index, nvmlDevice_t* device )
    Parameters
        index
            The index of the target GPU, >= 0 and < accessibleDevices
        device
            Reference in which to return the device handle
    Returns
        NVML_SUCCESS if device has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully initialized
        NVML_ERROR_INVALID_ARGUMENT if index is invalid or device is NULL
        NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
        NVML_ERROR_NO_PERMISSION if the user doesn't have permission to talk to this device
        NVML_ERROR_IRQ_ISSUE if NVIDIA kernel detected an interrupt issue with the attached GPUs
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Acquire the handle for a particular device, based on its index.

        For all products.

        Valid indices are derived from the accessibleDevices count returned by nvmlDeviceGetCount_v2(). For example, if accessibleDevices is 2 the
        valid indices are 0 and 1, corresponding to GPU 0 and GPU 1.

        The order in which NVML enumerates devices has no guarantees of consistency between reboots. For that reason it is recommended that devices
        be looked up by their PCI ids or UUID. See nvmlDeviceGetHandleByUUID() and nvmlDeviceGetHandleByPciBusId_v2().

        Note: The NVML index may not correlate with other APIs, such as the CUDA device index.

        Starting from NVML 5, this API causes NVML to initialize the target GPU NVML may initialize additional GPUs if:
            The target GPU is an SLI slave

            Note: New nvmlDeviceGetCount_v2 (default in NVML 5.319) returns count of all devices in the system even if nvmlDeviceGetHandleByIndex_v2
            returns NVML_ERROR_NO_PERMISSION for such device. Update your code to handle this error, or use NVML 4.304 or older nvml header file.
            For backward binary compatibility reasons _v1 version of the API is still present in the shared library. Old _v1 version of
            nvmlDeviceGetCount doesn't count devices that NVML has no permission to talk to.
            
            This means that nvmlDeviceGetHandleByIndex_v2 and _v1 can return different devices for the same index. If you don't touch macros that map
            old (_v1) versions to _v2 versions at the top of the file you don't need to worry about that.

        See also:
            nvmlDeviceGetIndex
            nvmlDeviceGetCount
  */
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetHandleByIndex_v2_t)(unsigned int, nvmlDevice_t*);
  /* Updated: 2025-12-09
    nvmlReturn_t nvmlDeviceGetHandleByUUID ( const char* uuid, nvmlDevice_t* device )
    Parameters
        uuid
            The UUID of the target GPU or MIG instance
        device
            Reference in which to return the device handle or MIG device handle
    Returns
        NVML_SUCCESS if device has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully initialized
        NVML_ERROR_INVALID_ARGUMENT if uuid is invalid or device is null
        NVML_ERROR_NOT_FOUND if uuid does not match a valid device on the system
        NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
        NVML_ERROR_IRQ_ISSUE if NVIDIA kernel detected an interrupt issue with the attached GPUs
        NVML_ERROR_GPU_IS_LOST if any GPU has fallen off the bus or is otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Acquire the handle for a particular device, based on its globally unique immutable UUID (in ASCII format) associated with each device.

        For all products.

        Starting from NVML 5, this API causes NVML to initialize the target GPU NVML may initialize additional GPUs as it searches for the target GPU

        See also:
            nvmlDeviceGetUUID
  */
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetHandleByUUID_t)(const char*, nvmlDevice_t*);
  /* Updated: 2025-12-09
    nvmlReturn_t nvmlDeviceGetUUID ( nvmlDevice_t device, char* uuid, unsigned int  length )
    Parameters
        device
        The identifier of the target device
        uuid
            Reference in which to return the GPU UUID
        length
            The maximum allowed length of the string returned in uuid
    Returns
        NVML_SUCCESS if uuid has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid, or uuid is NULL
        NVML_ERROR_INSUFFICIENT_SIZE if length is too small
        NVML_ERROR_NOT_SUPPORTED if the device does not support this feature
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the globally unique immutable UUID associated with this device, as a 5 part hexadecimal string, that augments the immutable,
        board serial identifier.

        For all products.

        The UUID is a globally unique identifier. It is the only available identifier for pre-Fermi-architecture products. It does NOT correspond
        to any identifier printed on the board. It will not exceed 96 characters in length (including the NULL terminator).
        See nvmlConstants::NVML_DEVICE_UUID_V2_BUFFER_SIZE.

        When used with MIG device handles the API returns globally unique UUIDs which can be used to identify MIG devices across both GPU and MIG
        devices. UUIDs are immutable for the lifetime of a MIG device.
  */
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetUUID_t)(nvmlDevice_t, char*, unsigned int);
  /* Updated: 2025-12-09
    nvmlReturn_t nvmlDeviceGetCudaComputeCapability ( nvmlDevice_t device, int* major, int* minor )
    Parameters
        device
            The identifier of the target device
        major
            Reference in which to return the major CUDA compute capability
        minor
            Reference in which to return the minor CUDA compute capability
    Returns
        NVML_SUCCESS if major and minor have been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or major or minor are NULL
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the CUDA compute capability of the device.

        For all products.

        Returns the major and minor compute capability version numbers of the device. The major and minor versions are equivalent to the
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR and CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR attributes that would be returned
        by CUDA's cuDeviceGetAttribute().
  */
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetCudaComputeCapability_t)(nvmlDevice_t, int*, int*);
  /* Updated: 2025-12-09
    nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed ( nvmlDevice_t device, unsigned int* maxSpeed )
    Parameters
        device
            The identifier of the target device
        maxSpeed
            The devices's PCIE Max Link speed in MBPS
    Returns
        NVML_SUCCESS if PCIe Max Link Speed is successfully retrieved
        NVML_ERROR_UNINITIALIZED if the library has not been successfully initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid, or maxSpeed is NULL
        NVML_ERROR_NOT_SUPPORTED if this query is not supported by the device
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is otherwise inaccessible
    Description
        Gets the device's PCIE Max Link speed in MBPS
  */
  typedef nvmlReturn_t (NVML_API *nvmlDeviceGetPciInfo_v3_t)(nvmlDevice_t, nvmlPciInfo_t*);
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
