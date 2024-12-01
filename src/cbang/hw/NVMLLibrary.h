#pragma once

#include "ComputeDevice.h"
#include "GPUMeasurement.h"

#include <cbang/os/DynamicLibrary.h>
#include <cbang/util/Singleton.h>

#include <vector>

namespace cb {
  class NVMLLibrary : public DynamicLibrary,
                      public Singleton<NVMLLibrary> {
    typedef std::vector<ComputeDevice> devices_t;
    devices_t devices;

  public:
    NVMLLibrary(Inaccessible);
    ~NVMLLibrary() override;

    inline static const char *getName() {return "NVML";}
    unsigned getDeviceCount() const {return devices.size();}
    const ComputeDevice &getDevice(unsigned i) const;

    typedef devices_t::const_iterator iterator;
    iterator begin() const {return devices.begin();}
    iterator end() const {return devices.end();}

    bool tryGetMeasurements(const char* uuid, GPUMeasurement &measurements);
  };
}
