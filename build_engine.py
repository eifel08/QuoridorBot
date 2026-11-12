"""Compile the C++ search engine (engine/) with the installed C++ compiler.

    python build_engine.py

Tries g++, clang++, c++ and MSVC's cl (in a "Developer Command Prompt") and
writes engine/build/quoridor_engine (.exe on Windows). Alternatives:
``make -C engine`` or CMake, see engine/README.md.
"""

import sys

from quoridor.bots.cpp_engine import EngineError, build

if __name__ == "__main__":
    try:
        print("built", build())
    except EngineError as exc:
        sys.exit(str(exc))
