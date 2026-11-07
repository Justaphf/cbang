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

#pragma once

#include <vector>


namespace cb {
  // Live GPU telemetry.  Values the device does not report are left at their
  // defaults: zero, or -1 for the P-state and utilizations, where 0 is valid.
  struct GPUMeasurement {
    unsigned gpuClock         = 0;  // MHz
    unsigned gpuClockMax      = 0;  // MHz
    unsigned memClock         = 0;  // MHz
    unsigned memClockMax      = 0;  // MHz
    double   power            = 0;  // Watts
    double   powerLimit       = 0;  // Watts, enforced
    unsigned temperature      = 0;  // Celsius, GPU die
    int      utilization      = -1; // Percent of time a kernel ran
    int      memUtilization   = -1; // Percent of time memory was accessed
    int      pstate           = -1; // 0 is maximum performance
    unsigned pcieGen          = 0;
    unsigned pcieGenMax       = 0;  // Possible with this device and system
    unsigned pcieGenDeviceMax = 0;  // Supported by the device alone
    unsigned pcieWidth        = 0;  // Lanes
    unsigned pcieWidthMax     = 0;  // Lanes
    std::vector<unsigned> fans;     // Percent, per fan
  };
}
