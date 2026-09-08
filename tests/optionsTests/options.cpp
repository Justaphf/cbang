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


#include <cbang/Exception.h>
#include <cbang/Catch.h>
#include <cbang/config/Options.h>
#include <cbang/config/OptionProxy.h>
#include <cbang/xml/Reader.h>

#include <iostream>
#include <sstream>

using namespace cb;
using namespace std;


namespace {
  void usage(const char *name) {
    cout
      << "Usage: " << name << " [OPTIONS]\n\n"
      << "Reads an XML fragment into an OptionProxy and reports where each\n"
      << "option was written: into the proxy, or through to its parent.\n\n"
      << "OPTIONS:\n"
      << "\t--help         Print this help screen and exit.\n"
      << "\t--xml <text>   Read options from the given XML text.\n"
      << endl;
  }


  string value(const OptionPtr &option) {
    if (option.isNull())      return "<missing>";
    if (!option->hasValue())  return "<unset>";
    return option->toString();
  }


  void report(const char *name, Options &parent, Options &proxy) {
    cout
      << name << ": local=" << (proxy.local(name) ? "yes" : "no")
      << " proxy=" << value(proxy.get(name))
      << " parent=" << value(parent.get(name))
      << " parent-set=" << (parent.get(name)->isSet() ? "yes" : "no")
      << endl;
  }
}


int main(int argc, char *argv[]) {
  try {
    string xml;

    for (int i = 1; i < argc; i++) {
      string arg = argv[i];

      if (arg == "--help") {usage(argv[0]); return 0;}

      else if (arg == "--xml" && i < argc - 1) xml = argv[++i];

      else {
        usage(argv[0]);
        return 1;
      }
    }

    // Options as an application declares them
    Options parent;
    parent.add("flag",  "A boolean option.")->setDefault(false);
    parent.add("other", "Another boolean option.")->setDefault(false);
    parent.add("name",  "A string option.")->setDefault("");

    // A sub-scope, as a project or other nested config section uses.  Writing
    // through the proxy must not modify the parent's Option.
    OptionProxy proxy(parent);

    istringstream stream(xml);
    XML::Reader reader;
    reader.pushFile("<test>"); // Reading a stream still needs a file name
    reader.read(stream, &proxy);

    report("flag",  parent, proxy);
    report("other", parent, proxy);
    report("name",  parent, proxy);

    return 0;
  } CATCH_ERROR;

  return 1;
}
