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

#include <cbang/String.h>
#include <cbang/util/Regex.h>

#include <cbang/Catch.h>

#include <iostream>

using namespace std;
using namespace cb;


int usage(const char *name) {
  cerr << "Usage: " << name << " <-r | -s> <pattern> <replace> <subject>\n"
       << "  -r  Regex::replace()\n"
       << "  -s  String::replace()\n";
  return 1;
}


int main(int argc, char *argv[]) {
  try {
    if (argc != 5) return usage(argv[0]);

    string pattern = argv[2];
    string replace = argv[3];
    string subject = argv[4];

    if (string("-r") == argv[1])
      cout << Regex(pattern).replace(subject, replace) << endl;

    else if (string("-s") == argv[1])
      cout << String::replace(subject, pattern, replace) << endl;

    else return usage(argv[0]);

    return 0;

  } CBANG_CATCH_ERROR;
  return 1;
}
