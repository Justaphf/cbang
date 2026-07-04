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

// Logger line-ending regression test.
//
// Exercises the OS x logCRLF terminator matrix for the Logger's file and
// screen sinks, plus normalization of CR within logged content:
//
//   | OS      | logCRLF | console (captured) | log file |
//   |---------|---------|--------------------|----------|
//   | Windows | true    | \r\n               | \r\n     |
//   | Windows | false   | \r\n               | \n       |
//   | Linux   | true    | \r\n               | \r\n     |
//   | Linux   | false   | \n                 | \n       |
//
// Golden selection: log file -> logCRLF ? crlfTRUE.bin : crlfFALSE.bin on
// both OSes; console -> always crlfTRUE.bin on Windows (the text-mode CRT
// translates every '\n' to "\r\n"), else the same as the file.  The console
// is captured through a real text-mode std::ofstream so that CRT translation
// is reproduced -- an in-process stringstream would not translate and would
// falsely pass the (Windows, logCRLF=false, console) cell.

#include <cbang/log/Logger.h>
#include <cbang/os/SystemUtilities.h>
#include <cbang/Catch.h>

#include <string>
#include <fstream>
#include <iostream>

using namespace std;
using namespace cb;


namespace {
  // Read an entire file as raw bytes (no text translation).
  string readBytes(const string &path) {
    ifstream in(path.c_str(), ios::binary);
    if (!in) THROW("Failed to open '" << path << "'");
    return string((istreambuf_iterator<char>(in)),
                  istreambuf_iterator<char>());
  }


  string escape(const string &s) {
    string out;
    for (char c : s) {
      if      (c == '\r') out += "\\r";
      else if (c == '\n') out += "\\n";
      else                out += c;
    }
    return out;
  }


  // Compare and report; returns true on match.
  bool expectBytes(const string &label, const string &got,
                   const string &want) {
    if (got == want) return true;
    cerr << "MISMATCH [" << label << "]\n"
         << "  expected: \"" << escape(want) << "\" (" << want.size()
         << " bytes)\n"
         << "  actual:   \"" << escape(got)  << "\" (" << got.size()
         << " bytes)\n";
    return false;
  }


  // Reset the Logger's sinks to the given fresh streams for one cell.  Fixed
  // formatting (no header/time/color) so a logged line is exactly its content
  // plus the per-sink terminator.
  void configureLogger(bool logCRLF, const SmartPointer<ostream> &screen,
                       const string &logPath) {
    Logger &log = Logger::instance();
    log.setLogHeader(false);
    log.setLogTime(false);
    log.setLogDate(false);
    log.setLogColor(false);
    log.setLogToScreen(true);
    log.setLogTruncate(true);   // deterministic file, no append/rotate carry
    log.setLogRotate(false);
    log.setLogCRLF(logCRLF);
    log.setScreenStream(screen);
    log.startLogFile(logPath);  // writes the "Log Started" bar to the FILE only
  }


  // Log each LF-separated line of `content` as its own raw line (LOG_RAW
  // suppresses the per-line header; the terminator is appended by the Logger
  // per logCRLF / per sink).  Trailing partial line (no final '\n') is ignored;
  // callers pass fully terminated content.
  void logLines(const string &content) {
    string line;
    for (char c : content) {
      if (c == '\n') {LOG_RAW(line); line.clear();}
      else line += c;
    }
  }
}


// One matrix cell: write `input` through the Logger to a fresh log file and a
// fresh text-mode screen capture, then compare each against its golden.
static bool runCell(const string &suiteDir, bool logCRLF,
                    const string &input, const string &fileGolden,
                    const string &screenGolden) {
  string tag = string("logCRLF=") + (logCRLF ? "true" : "false");

  string logPath    = "log-"    + string(logCRLF ? "true" : "false") + ".txt";
  string screenPath = "screen-" + string(logCRLF ? "true" : "false") + ".txt";

  streampos barLen = 0;

  {
    // Text-mode ofstream (NO ios::binary): on Windows its CRT reproduces the
    // console '\n'->"\r\n" translation, giving a faithful screen capture.
    SmartPointer<ostream> screen =
      new ofstream(screenPath.c_str(), ios::out | ios::trunc);
    if (!*screen) THROW("Failed to open screen capture '" << screenPath << "'");

    configureLogger(logCRLF, screen, logPath);

    // The log file already holds the flushed "Log Started" bar; record its
    // length so we compare only the bytes written AFTER it (robust to any bar
    // format change -- do NOT compare from byte 0 or line-count past the bar).
    {
      ifstream f(logPath.c_str(), ios::binary | ios::ate);
      barLen = f.tellg();
    }

    logLines(input);

    // Drop the screen stream before reading the capture back so the ofstream
    // is flushed/closed.
    Logger::instance().setScreenStream(cout);
  }

  bool ok = true;

  // Log-file cell: bytes after the startup bar vs the selected golden.
  string fileBytes = readBytes(logPath);
  if (barLen < 0 || (size_t)barLen > fileBytes.size()) {
    cerr << "Bad bar offset " << barLen << " for '" << logPath << "'\n";
    ok = false;
  } else {
    string fileContent = fileBytes.substr((size_t)barLen);
    ok &= expectBytes(tag + " file", fileContent, fileGolden);
  }

  // Console cell: the text-mode capture vs the selected golden.
  string screenBytes = readBytes(screenPath);
  ok &= expectBytes(tag + " screen", screenBytes, screenGolden);

  // No matrix cell may double the console terminator.
  if (screenBytes.find("\r\r\n") != string::npos) {
    cerr << "DOUBLING [" << tag << " screen]: found \"\\r\\r\\n\"\n";
    ok = false;
  }

  return ok;
}


// Embedded & trailing CR within logged content: LogStream drops every content
// '\r' before the terminator is appended, so both sinks see the content with
// CR removed, followed by the normal per-cell terminator, and the console
// never doubles.  The CR-bearing input is built in-source to keep it out of
// autocrlf's reach.
static bool runCRCase(bool logCRLF) {
  string tag = string("CR-case logCRLF=") + (logCRLF ? "true" : "false");

  // Embedded ("a\rb") and trailing ("c\r") CR, one line each.
  const char *lines[] = {"a\rb", "c\r"};

  string logPath    = "crlog-"    + string(logCRLF ? "true" : "false") + ".txt";
  string screenPath = "crscreen-" + string(logCRLF ? "true" : "false") + ".txt";

  // Expected: content '\r' removed -> "ab", "c"; each followed by the per-cell
  // terminator.  File terminator = logCRLF ? "\r\n" : "\n".  Console terminator
  // on Windows is always "\r\n" (text-mode CRT); on Linux it matches the file.
  string fileTerm = logCRLF ? "\r\n" : "\n";
#ifdef _WIN32
  string screenTerm = "\r\n";
#else
  string screenTerm = fileTerm;
#endif
  string expectFile   = "ab" + fileTerm   + "c" + fileTerm;
  string expectScreen = "ab" + screenTerm + "c" + screenTerm;

  streampos barLen = 0;

  {
    SmartPointer<ostream> screen =
      new ofstream(screenPath.c_str(), ios::out | ios::trunc);
    if (!*screen) THROW("Failed to open screen capture '" << screenPath << "'");

    configureLogger(logCRLF, screen, logPath);

    {
      ifstream f(logPath.c_str(), ios::binary | ios::ate);
      barLen = f.tellg();
    }

    for (const char *l : lines) LOG_RAW(l);

    Logger::instance().setScreenStream(cout);
  }

  bool ok = true;

  string fileBytes = readBytes(logPath);
  if (barLen < 0 || (size_t)barLen > fileBytes.size()) {
    cerr << "Bad bar offset " << barLen << " for '" << logPath << "'\n";
    ok = false;
  } else ok &= expectBytes(tag + " file", fileBytes.substr((size_t)barLen),
                           expectFile);

  string screenBytes = readBytes(screenPath);
  ok &= expectBytes(tag + " screen", screenBytes, expectScreen);

  if (screenBytes.find("\r\r\n") != string::npos) {
    cerr << "DOUBLING [" << tag << " screen]: found \"\\r\\r\\n\"\n";
    ok = false;
  }

  return ok;
}


int main(int argc, char *argv[]) {
  Exception::printLocations    = false;
  Exception::enableStackTraces = false;

  try {
    if (argc != 2) {
      cerr << "Usage: " << argv[0] << " <suite-dir>\n";
      return 2;
    }

    string suiteDir = argv[1];

    // The input content is the LF golden read as raw bytes.
    string input        = readBytes(suiteDir + "/crlfFALSE.bin");
    string crlfFalseGold = input;                                  // LF golden
    string crlfTrueGold  = readBytes(suiteDir + "/crlfTRUE.bin");  // CRLF golden

#ifdef _WIN32
    // Windows text-mode console is always "\r\n" regardless of logCRLF.
    string screenTrue  = crlfTrueGold;
    string screenFalse = crlfTrueGold;
#else
    string screenTrue  = crlfTrueGold;
    string screenFalse = crlfFalseGold;
#endif

    bool ok = true;

    // logCRLF = true:  file -> CRLF golden; console -> CRLF golden (both OSes).
    ok &= runCell(suiteDir, true,  input, crlfTrueGold,  screenTrue);
    // logCRLF = false: file -> LF golden;  console -> per table above.
    ok &= runCell(suiteDir, false, input, crlfFalseGold, screenFalse);

    // v2 / A6: content-CR normalization for both modes.
    ok &= runCRCase(true);
    ok &= runCRCase(false);

    return ok ? 0 : 1;
  } CBANG_CATCH_ERROR;

  return 1;
}
