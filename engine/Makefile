# Simple build without CMake:  make  ->  build/quoridor_engine
#   make baseline  ->  build/quoridor_engine_baseline (frozen main engine, the tournament benchmark)
CXX ?= g++
CXXFLAGS ?= -O3 -std=c++20 -Wall -Wextra -DNDEBUG
NATIVE ?= -march=native -flto
SRC = src/board.cpp src/search.cpp src/main.cpp
HDR = src/board.h src/search.h
BASE_SRC = baseline/board.cpp baseline/search.cpp baseline/main.cpp

build/quoridor_engine: $(SRC) $(HDR)
	mkdir -p build
	$(CXX) $(CXXFLAGS) $(NATIVE) $(SRC) -o $@

baseline: build/quoridor_engine_baseline

build/quoridor_engine_baseline: $(BASE_SRC) baseline/board.h baseline/search.h
	mkdir -p build
	$(CXX) -O3 -std=c++20 -DNDEBUG $(BASE_SRC) -o $@

clean:
	rm -rf build

.PHONY: clean baseline
