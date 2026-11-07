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

  /* Updated: 2026-11-07
    nvmlUtilization_t
    Description
        Utilization information for a device. Each sample period may be
        between 1 second and 1/6 second, depending on the product being
        queried.
    Members
        gpu
            Percent of time over the past sample period during which one or
            more kernels was executing on the GPU.
        memory
            Percent of time over the past sample period during which global
            (device) memory was being read or written.
  */
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

  /* NVML API reference: https://docs.nvidia.com/deploy/nvml-api/ */
  /* Initialization and clean-up:
     api/group__nvmlInitializationAndCleanup.html */
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlInit_v2 ( void )
    Returns
        NVML_SUCCESS if the NVML Library was properly initialized.
        NVML_ERROR_DRIVER_NOT_LOADED if the NVIDIA driver is not running.
        NVML_ERROR_NO_PERMISSION if the NVML Library does not have permission
            to talk to the driver.
        NVML_ERROR_UNKNOWN if there is an unexpected error.
    Description
        Initialize the NVML Library lazily, without allocating any device state.

        This will initialize the NVML Library state without enumerating any
        discovered devices. This will allow NVML to communicate with a device,
        even if other devices are in an unstable or bad state. Enumeration of a
        device can be done by obtaining the device handle via the
        nvmlDeviceGetHandleBy* class of APIs.

        This method needs to be called once before any usage of NVML Library
        APIs.

        For all products.

        See also:
            nvmlShutdown()

        Note: A reference count of the number of initializations is maintained,
        and a corresponding call to nvmlShutdown() needs to be issued once usage
        of the NVML Library is complete. Shutdown will only occur after the
        reference count reaches zero.
  */
  using nvmlInit_v2_t = int (*)();
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlShutdown ( void )
    Returns
        NVML_SUCCESS if the NVML Library was properly shut down.
        NVML_ERROR_UNINITIALIZED if the NVML Library was not previously
            initialized.
        NVML_ERROR_UNKNOWN if there is an unexpected error.
    Description
        Shut down and cleanup NVML Library state.

        This will shut down and cleanup NVML Library state by releasing all
        device and library resources previously allocated with nvmlInit_v2() or
        nvmlInitWithFlags(). This should be called after all NVML work is done,
        and once for each call to nvmlInit_v2() or nvmlInitWithFlags().

        Complete shutdown will only occur when the reference count of all prior
        NVML initializations reaches zero. No error will be reported if this is
        called more times than nvmlInit_v2() or nvmlInitWithFlags().

        For all products.
  */
  using nvmlShutdown_t = int (*)();
  /* System queries: api/group__nvmlSystemQueries.html */
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2 ( int* cudaDriverVersion )
    Parameters
        cudaDriverVersion
            Reference in which to return the version identifier
    Returns
        NVML_SUCCESS if cudaDriverVersion has been set
        NVML_ERROR_INVALID_ARGUMENT if cudaDriverVersion is NULL
        NVML_ERROR_LIBRARY_NOT_FOUND if libcuda.so.1 or libcuda.dll is not found
        NVML_ERROR_FUNCTION_NOT_FOUND if cuDriverGetVersion() is not found in
            the shared library
    Description
        Retrieves the version of the CUDA driver from the shared library.

        For all products.

        The returned CUDA driver version by calling cuDriverGetVersion()
  */
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlSystemGetCudaDriverVersion ( int* cudaDriverVersion )
    Parameters
        cudaDriverVersion
            Reference in which to return the version identifier
    Returns
        NVML_SUCCESS if cudaDriverVersion has been set
        NVML_ERROR_INVALID_ARGUMENT if cudaDriverVersion is NULL
    Description
        Retrieves the version of the CUDA driver.

        For all products.

        The CUDA driver version returned will be retrieved from the currently
        installed version of CUDA. If the cuda library is not found, this
        function will return a known supported version number.
  */
  using nvmlSystemGetCudaDriverVersion_v2_t = int (*)(int *);
  using nvmlSystemGetCudaDriverVersion_t = int (*)(int *);
  using nvmlSystemGetDriverVersion_t = int (*)(char *, unsigned);
  /* Device queries: api/group__nvmlDeviceQueries.html */
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetCount_v2 ( unsigned int* deviceCount )
    Parameters
        deviceCount
            Reference in which to return the number of accessible devices
    Returns
        NVML_SUCCESS if deviceCount has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if deviceCount is NULL
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the number of compute devices in the system. A compute device
        is a single GPU.

        For all products.

        Note: New nvmlDeviceGetCount_v2 (default in NVML 5.319) returns count of
        all devices in the system even if nvmlDeviceGetHandleByIndex_v2 returns
        NVML_ERROR_NO_PERMISSION for such device. Update your code to handle
        this error, or use NVML 4.304 or older nvml header file. For backward
        binary compatibility reasons _v1 version of the API is still present in
        the shared library. Old _v1 version of nvmlDeviceGetCount doesn't count
        devices that NVML has no permission to talk to.
  */
  using nvmlDeviceGetCount_v2_t = int (*)(unsigned *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetHandleByIndex_v2 ( unsigned int  index,
        nvmlDevice_t* device )
    Parameters
        index
            The index of the target GPU, >= 0 and < accessibleDevices
        device
            Reference in which to return the device handle
    Returns
        NVML_SUCCESS if device has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if index is invalid or device is NULL
        NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly
            attached external power cables
        NVML_ERROR_NO_PERMISSION if the user doesn't have permission to talk to
            this device
        NVML_ERROR_IRQ_ISSUE if NVIDIA kernel detected an interrupt issue with
            the attached GPUs
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Acquire the handle for a particular device, based on its index.

        For all products.

        Valid indices are derived from the accessibleDevices count returned by
        nvmlDeviceGetCount_v2(). For example, if accessibleDevices is 2 the
        valid indices are 0 and 1, corresponding to GPU 0 and GPU 1.

        The order in which NVML enumerates devices has no guarantees of
        consistency between reboots. For that reason it is recommended that
        devices be looked up by their PCI ids or UUID. See
        nvmlDeviceGetHandleByUUID() and nvmlDeviceGetHandleByPciBusId_v2().

        Note: The NVML index may not correlate with other APIs, such as the CUDA
        device index.

        Starting from NVML 5, this API causes NVML to initialize the target GPU
        NVML may initialize additional GPUs if:
            The target GPU is an SLI slave

            Note: New nvmlDeviceGetCount_v2 (default in NVML 5.319) returns
            count of all devices in the system even if
            nvmlDeviceGetHandleByIndex_v2 returns NVML_ERROR_NO_PERMISSION for
            such device. Update your code to handle this error, or use NVML
            4.304 or older nvml header file. For backward binary compatibility
            reasons _v1 version of the API is still present in the shared
            library. Old _v1 version of nvmlDeviceGetCount doesn't count devices
            that NVML has no permission to talk to.

            This means that nvmlDeviceGetHandleByIndex_v2 and _v1 can return
            different devices for the same index. If you don't touch macros that
            map old (_v1) versions to _v2 versions at the top of the file you
            don't need to worry about that.

        See also:
            nvmlDeviceGetIndex
            nvmlDeviceGetCount
  */
  using nvmlDeviceGetHandleByIndex_v2_t = int (*)(unsigned, nvmlDevice_t *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetHandleByUUID ( const char* uuid,
        nvmlDevice_t* device )
    Parameters
        uuid
            The UUID of the target GPU or MIG instance
        device
            Reference in which to return the device handle or MIG device handle
    Returns
        NVML_SUCCESS if device has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if uuid is invalid or device is null
        NVML_ERROR_NOT_FOUND if uuid does not match a valid device on the system
        NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly
            attached external power cables
        NVML_ERROR_IRQ_ISSUE if NVIDIA kernel detected an interrupt issue with
            the attached GPUs
        NVML_ERROR_GPU_IS_LOST if any GPU has fallen off the bus or is otherwise
            inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Acquire the handle for a particular device, based on its globally unique
        immutable UUID (in ASCII format) associated with each device.

        For all products.

        Starting from NVML 5, this API causes NVML to initialize the target GPU
        NVML may initialize additional GPUs as it searches for the target GPU

        See also:
            nvmlDeviceGetUUID
  */
  using nvmlDeviceGetHandleByUUID_t = int (*)(const char *, nvmlDevice_t *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetUUID ( nvmlDevice_t device, char* uuid,
        unsigned int  length )
    Parameters
        device
            The identifier of the target device
        uuid
            Reference in which to return the GPU UUID
        length
            The maximum allowed length of the string returned in uuid
    Returns
        NVML_SUCCESS if uuid has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid, or uuid is NULL
        NVML_ERROR_INSUFFICIENT_SIZE if length is too small
        NVML_ERROR_NOT_SUPPORTED if the device does not support this feature
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the globally unique immutable UUID associated with this
        device, as a 5 part hexadecimal string, that augments the immutable,
        board serial identifier.

        For all products.

        The UUID is a globally unique identifier. It is the only available
        identifier for pre-Fermi-architecture products. It does NOT correspond
        to any identifier printed on the board. It will not exceed 96 characters
        in length (including the NULL terminator). See
        nvmlConstants::NVML_DEVICE_UUID_V2_BUFFER_SIZE.

        When used with MIG device handles the API returns globally unique UUIDs
        which can be used to identify MIG devices across both GPU and MIG
        devices. UUIDs are immutable for the lifetime of a MIG device.
  */
  using nvmlDeviceGetUUID_t = int (*)(nvmlDevice_t, char *, unsigned);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetName ( nvmlDevice_t device, char* name,
        unsigned int  length )
    Parameters
        device
            The identifier of the target device
        name
            Reference in which to return the product name
        length
            The maximum allowed length of the string returned in name
    Returns
        NVML_SUCCESS if name has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid, or name is NULL
        NVML_ERROR_INSUFFICIENT_SIZE if length is too small
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the name of this device.

        For all products.

        The name is an alphanumeric string that denotes a particular product,
        e.g. Tesla C2070. It will not exceed 96 characters in length (including
        the NULL terminator). See
        nvmlConstants::NVML_DEVICE_NAME_V2_BUFFER_SIZE.

        When used with MIG device handles the API returns MIG device names which
        can be used to identify devices based on their attributes.
  */
  using nvmlDeviceGetName_t = int (*)(nvmlDevice_t, char *, unsigned);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetCudaComputeCapability ( nvmlDevice_t device,
        int* major, int* minor )
    Parameters
        device
            The identifier of the target device
        major
            Reference in which to return the major CUDA compute capability
        minor
            Reference in which to return the minor CUDA compute capability
    Returns
        NVML_SUCCESS if major and minor have been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or major or minor are
            NULL
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the CUDA compute capability of the device.

        For all products.

        Returns the major and minor compute capability version numbers of the
        device. The major and minor versions are equivalent to the
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR and
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR attributes that would be
        returned by CUDA's cuDeviceGetAttribute().
  */
  using nvmlDeviceGetCudaComputeCapability_t =
    int (*)(nvmlDevice_t, int *, int *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetPciInfo_v3 ( nvmlDevice_t device,
        nvmlPciInfo_t* pci )
    Parameters
        device
            The identifier of the target device
        pci
            Reference in which to return the PCI info
    Returns
        NVML_SUCCESS if pci has been populated
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or pci is NULL
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the PCI attributes of this device.

        For all products.

        See nvmlPciInfo_t for details on the available PCI info.
  */
  using nvmlDeviceGetPciInfo_v3_t = int (*)(nvmlDevice_t, nvmlPciInfo_t *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetClock ( nvmlDevice_t device,
        nvmlClockType_t clockType, nvmlClockId_t clockId,
        unsigned int* clockMHz )
    Parameters
        device
            The identifier of the target device
        clockType
            Identify which clock domain to query
        clockId
            Identify which clock in the domain to query
        clockMHz
            Reference in which to return the clock in MHz
    Returns
        NVML_SUCCESS if clockMHz has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or clockMHz is NULL or
            clockType is invalid
        NVML_ERROR_NOT_SUPPORTED if the device does not support this feature
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the clock speed for the clock specified by the clock type and
        clock ID.

        For Kepler or newer fully supported devices.
  */
  using nvmlDeviceGetClock_t = int (*)(nvmlDevice_t, int, int, unsigned *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetMaxClockInfo ( nvmlDevice_t device,
        nvmlClockType_t type, unsigned int* clock )
    Parameters
        device
            The identifier of the target device
        type
            Identify which clock domain to query
        clock
            Reference in which to return the clock speed in MHz
    Returns
        NVML_SUCCESS if clock has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or clock is NULL
        NVML_ERROR_NOT_SUPPORTED if the device cannot report the specified clock
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the maximum clock speeds for the device.

        For Fermi or newer fully supported devices.

        See nvmlClockType_t for details on available clock information.

        Note:
        Current P0 clocks (reported by nvmlDeviceGetClockInfo) can differ from
        max clocks by a few MHz.
  */
  using nvmlDeviceGetMaxClockInfo_t = int (*)(nvmlDevice_t, int, unsigned *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetPowerUsage ( nvmlDevice_t device,
        unsigned int* power )
    Parameters
        device
            The identifier of the target device
        power
            Reference in which to return the power usage information
    Returns
        NVML_SUCCESS if power has been populated
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or power is NULL
        NVML_ERROR_NOT_SUPPORTED if the device does not support power readings
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves power usage for this GPU in milliwatts and its associated
        circuitry (e.g. memory)

        For Fermi or newer fully supported devices.

        On Fermi and Kepler GPUs the reading is accurate to within +/- 5% of
        current power draw. On Ampere (except GA100) or newer GPUs, the API
        returns power averaged over 1 sec interval. On GA100 and older
        architectures, instantaneous power is returned.

        See NVML_FI_DEV_POWER_AVERAGE and NVML_FI_DEV_POWER_INSTANT to query
        specific power values.

        It is only available if power management mode is supported. See
        nvmlDeviceGetPowerManagementMode.
  */
  using nvmlDeviceGetPowerUsage_t = int (*)(nvmlDevice_t, unsigned *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit ( nvmlDevice_t device,
        unsigned int* limit )
    Parameters
        device
            The device to communicate with
        limit
            Reference in which to return the power management limit in
            milliwatts
    Returns
        NVML_SUCCESS if limit has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or limit is NULL
        NVML_ERROR_NOT_SUPPORTED if the device does not support this feature
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Get the effective power limit that the driver enforces after taking into
        account all limiters

        Note: This can be different from the nvmlDeviceGetPowerManagementLimit
        if other limits are set elsewhere This includes the out of band power
        limit interface

        For Kepler or newer fully supported devices.
  */
  using nvmlDeviceGetEnforcedPowerLimit_t = int (*)(nvmlDevice_t, unsigned *);
  /* Updated: 2026-11-07 (deprecation stub on the live page; parameters and
     returns as last published)
    nvmlReturn_t nvmlDeviceGetTemperature ( nvmlDevice_t device,
        nvmlTemperatureSensors_t sensorType, unsigned int* temp )
    Parameters
        device
            The identifier of the target device
        sensorType
            Flag that indicates which sensor reading to retrieve
        temp
            Reference in which to return the temperature reading
    Returns
        NVML_SUCCESS if temp has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid, sensorType is invalid
            or temp is NULL
        NVML_ERROR_NOT_SUPPORTED if the device does not have the specified
            sensor
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Deprecated: Use nvmlDeviceGetTemperatureV instead
  */
  using nvmlDeviceGetTemperature_t = int (*)(nvmlDevice_t, int, unsigned *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetTemperatureV ( nvmlDevice_t device,
        nvmlTemperature_t* temperature )
    Parameters
        device
            [in] Target device identifier.
        temperature
            [inout] Structure specifying the sensor type (input) and retrieved
            temperature value (output).
    Returns
        NVML_SUCCESS if temp has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid, sensorType is invalid
            or temp is NULL
        NVML_ERROR_NOT_SUPPORTED if the device does not have the specified
            sensor
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the current temperature readings (in degrees C) for the given
        device.

        For all products.

        Note: This API reports the current GPU temperature rounded to a whole
        degree Celsius. For the current GPU temperature at higher precision, use
        nvmlDeviceGetFieldValues with the NVML_FI_DEV_GPU_TEMP field, which
        returns the temperature in degrees C as a double value.
  */
  using nvmlDeviceGetTemperatureV_t =
    int (*)(nvmlDevice_t, nvmlTemperature_t *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetUtilizationRates ( nvmlDevice_t device,
        nvmlUtilization_t* utilization )
    Parameters
        device
            The identifier of the target device
        utilization
            Reference in which to return the utilization information
    Returns
        NVML_SUCCESS if utilization has been populated
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or utilization is NULL
        NVML_ERROR_NOT_SUPPORTED if the device does not support this feature
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the current utilization rates for the device's major
        subsystems.

        For Fermi or newer fully supported devices.

        See nvmlUtilization_t for details on available utilization rates.

        Note: During driver initialization when ECC is enabled one can see high
        GPU and Memory Utilization readings. This is caused by ECC Memory
        Scrubbing mechanism that is performed during driver initialization.

        Note: On MIG-enabled GPUs, querying device utilization rates is not
        currently supported.
  */
  using nvmlDeviceGetUtilizationRates_t =
    int (*)(nvmlDevice_t, nvmlUtilization_t *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetPerformanceState ( nvmlDevice_t device,
        nvmlPstates_t* pState )
    Parameters
        device
            The identifier of the target device
        pState
            Reference in which to return the performance state reading
    Returns
        NVML_SUCCESS if pState has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or pState is NULL
        NVML_ERROR_NOT_SUPPORTED if the device does not support this feature
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the current performance state for the device.

        For Fermi or newer fully supported devices.

        See nvmlPstates_t for details on allowed performance states.
  */
  using nvmlDeviceGetPerformanceState_t = int (*)(nvmlDevice_t, int *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration ( nvmlDevice_t device,
        unsigned int* currLinkGen )
    Parameters
        device
            The identifier of the target device
        currLinkGen
            Reference in which to return the current PCIe link generation
    Returns
        NVML_SUCCESS if currLinkGen has been populated
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or currLinkGen is null
        NVML_ERROR_NOT_SUPPORTED if PCIe link information is not available
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the current PCIe link generation

        For Fermi or newer fully supported devices.
  */
  using nvmlDeviceGetCurrPcieLinkGeneration_t =
    int (*)(nvmlDevice_t, unsigned *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration ( nvmlDevice_t device,
        unsigned int* maxLinkGen )
    Parameters
        device
            The identifier of the target device
        maxLinkGen
            Reference in which to return the max PCIe link generation
    Returns
        NVML_SUCCESS if maxLinkGen has been populated
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or maxLinkGen is null
        NVML_ERROR_NOT_SUPPORTED if PCIe link information is not available
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the maximum PCIe link generation possible with this device and
        system

        I.E. for a generation 2 PCIe device attached to a generation 1 PCIe bus
        the max link generation this function will report is generation 1.

        For Fermi or newer fully supported devices.
  */
  using nvmlDeviceGetMaxPcieLinkGeneration_t =
    int (*)(nvmlDevice_t, unsigned *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration ( nvmlDevice_t device,
        unsigned int* maxLinkGenDevice )
    Parameters
        device
            The identifier of the target device
        maxLinkGenDevice
            Reference in which to return the max PCIe link generation
    Returns
        NVML_SUCCESS if maxLinkGenDevice has been populated
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or maxLinkGenDevice is
            null
        NVML_ERROR_NOT_SUPPORTED if PCIe link information is not available
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the maximum PCIe link generation supported by this device

        For Fermi or newer fully supported devices.
  */
  using nvmlDeviceGetGpuMaxPcieLinkGeneration_t =
    int (*)(nvmlDevice_t, unsigned *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth ( nvmlDevice_t device,
        unsigned int* currLinkWidth )
    Parameters
        device
            The identifier of the target device
        currLinkWidth
            Reference in which to return the current PCIe link generation
    Returns
        NVML_SUCCESS if currLinkWidth has been populated
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or currLinkWidth is
            null
        NVML_ERROR_NOT_SUPPORTED if PCIe link information is not available
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the current PCIe link width

        For Fermi or newer fully supported devices.
  */
  using nvmlDeviceGetCurrPcieLinkWidth_t = int (*)(nvmlDevice_t, unsigned *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth ( nvmlDevice_t device,
        unsigned int* maxLinkWidth )
    Parameters
        device
            The identifier of the target device
        maxLinkWidth
            Reference in which to return the max PCIe link generation
    Returns
        NVML_SUCCESS if maxLinkWidth has been populated
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or maxLinkWidth is null
        NVML_ERROR_NOT_SUPPORTED if PCIe link information is not available
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the maximum PCIe link width possible with this device and
        system

        I.E. for a device with a 16x PCIe bus width attached to a 8x PCIe system
        bus this function will report a max link width of 8.

        For Fermi or newer fully supported devices.
  */
  using nvmlDeviceGetMaxPcieLinkWidth_t = int (*)(nvmlDevice_t, unsigned *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetNumFans ( nvmlDevice_t device,
        unsigned int* numFans )
    Parameters
        device
            The identifier of the target device
        numFans
            The number of fans
    Returns
        NVML_SUCCESS if fan number query was successful
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or numFans is NULL
        NVML_ERROR_NOT_SUPPORTED if the device does not have a fan
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the number of fans on the device.

        For all discrete products with dedicated fans.
  */
  using nvmlDeviceGetNumFans_t = int (*)(nvmlDevice_t, unsigned *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetFanSpeed_v2 ( nvmlDevice_t device,
        unsigned int  fan, unsigned int* speed )
    Parameters
        device
            The identifier of the target device
        fan
            The index of the target fan, zero indexed.
        speed
            Reference in which to return the fan speed percentage
    Returns
        NVML_SUCCESS if speed has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid, fan is not an
            acceptable index, or speed is NULL
        NVML_ERROR_NOT_SUPPORTED if the device does not have a fan or is newer
            than Maxwell
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the intended operating speed of the device's specified fan.

        Note: The reported speed is the intended fan speed. If the fan is
        physically blocked and unable to spin, the output will not match the
        actual fan speed.

        For all discrete products with dedicated fans.

        The fan speed is expressed as a percentage of the product's maximum
        noise tolerance fan speed. This value may exceed 100% in certain cases.
  */
  using nvmlDeviceGetFanSpeed_v2_t =
    int (*)(nvmlDevice_t, unsigned, unsigned *);
  /* Updated: 2026-11-07
    nvmlReturn_t nvmlDeviceGetFanSpeed ( nvmlDevice_t device,
        unsigned int* speed )
    Parameters
        device
            The identifier of the target device
        speed
            Reference in which to return the fan speed percentage
    Returns
        NVML_SUCCESS if speed has been set
        NVML_ERROR_UNINITIALIZED if the library has not been successfully
            initialized
        NVML_ERROR_INVALID_ARGUMENT if device is invalid or speed is NULL
        NVML_ERROR_NOT_SUPPORTED if the device does not have a fan
        NVML_ERROR_GPU_IS_LOST if the target GPU has fallen off the bus or is
            otherwise inaccessible
        NVML_ERROR_UNKNOWN on any unexpected error
    Description
        Retrieves the intended operating speed of the device's fan.

        Note: The reported speed is the intended fan speed. If the fan is
        physically blocked and unable to spin, the output will not match the
        actual fan speed.

        For all discrete products with dedicated fans.

        The fan speed is expressed as a percentage of the product's maximum
        noise tolerance fan speed. This value may exceed 100% in certain cases.
  */
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
