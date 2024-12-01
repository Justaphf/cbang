#pragma once

#include <cstdint>

namespace cb {
  struct GPUMeasurement {
    uint16_t gpuFreq_MHz = 0;         //!< The current GPU clock frequency in MHz
    uint16_t gpuFreqLimit_MHz = 0;    //!< The maximum GPU clock frequency (limit) in MHz
    uint16_t memFreq_MHz = 0;         //!< The current memory clock frequency in MHz
    uint16_t memFreqLimit_MHz = 0;    //!< The maximum memory clock frequency (limit) in MHz
    uint16_t currPower_Watts = 0;     //!< The current power usage reported by this device in Watts
    uint16_t maxPower_Watts = 0;      //!< The maximum power usage (limit) in Watts
    uint8_t gpuTemp_C = 0;            //!< The current GPU temperature in Celsius
    uint8_t pstate = 0;               //!< The current GPU P-state
    uint8_t currPCIeLinkWidth = 0;    //!< The current PCIe link width
    uint8_t maxPCIeLinkWidth = 0;     //!< The maximum PCIe link width possible with this device and system
    uint8_t currPCIeLinkGen = 0;      //!< The current PCIe link generation
    uint8_t maxPCIeLinkGen = 0;       //!< The maximum PCIe link generation possible with this device and system
    uint8_t maxPCIeLinkGenDevice = 0; //!< The maximum PCIe link generation supported by this device
    uint8_t fanCount = 0;             //!< The number of fans that can be independently addressed on the GPU
    uint8_t fan0Speed_pct = 0;        //!< The current fan 0 speed as a percent of the noise profile
    uint8_t fan1Speed_pct = 0;        //!< The current fan 1 speed as a percent of the noise profile
    uint8_t fan2Speed_pct = 0;        //!< The current fan 2 speed as a percent of the noise profile
  };
}
