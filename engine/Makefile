# Simple build without CMake:  make  ->  build/quoridor_engine
CXX ?= g++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -DNDEBUG
SRC = src/board.cpp src/search.cpp src/main.cpp
HDR = src/board.h src/search.h

build/quoridor_engine: $(SRC) $(HDR)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

clean:
	rm -rf build

.PHONY: clean
