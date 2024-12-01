#pragma once

#include <cstdint>

namespace cb {
  struct GPUMeasurement {
    uint16_t gpuFreq_MHz = 0;         //!< The current GPU clock frequency in MHz
    uint16_t gpuFreqLimit_MHz = 0;    //!< The maximum GPU clock frequency (limit) in MHz
    uint16_t memFreq_MHz = 0;         //!< The current memory clock frequency in MHz
    uint16_t memFreqLimit_MHz = 0;    //!< The maximum memory clock frequency (limit) in MHz
    uint8_t gpuTemp_C = 0;            //!< The current GPU temperature in Celsius
    uint8_t pstate = 0;               //!< The current GPU P-state
    uint8_t currPCIeLinkWidth = 0;    //!< The current PCIe link width
    uint8_t maxPCIeLinkWidth = 0;     //!< The maximum PCIe link width possible with this device and system
    uint8_t currPCIeLinkGen = 0;      //!< The current PCIe link generation
    uint8_t maxPCIeLinkGen = 0;       //!< The maximum PCIe link generation possible with this device and system
    uint8_t maxPCIeLinkGenDevice = 0; //!< The maximum PCIe link generation supported by this device
  };
}
