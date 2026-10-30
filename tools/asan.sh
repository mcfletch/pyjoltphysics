#!/bin/bash
# Build the module with AddressSanitizer and UndefinedBehaviorSanitizer and run
# the test suite against it.
#
# usage: tools/asan.sh BUILD_DIR [pytest arguments]
#
# Python is not built with the sanitizers, so their runtimes are preloaded.
# Leak checking is off: the interpreter keeps memory until exit by design.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
build="$1"
shift
python="$(command -v python)"
cmake -S "$root" -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPYJOLTPHYSICS_SANITIZE=ON -DPython_EXECUTABLE="$python" \
    -Dnanobind_DIR="$("$python" -c 'import nanobind; print(nanobind.cmake_dir())')"
cmake --build "$build" -j "${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
package="$build/package"
rm -rf "$package"
mkdir -p "$package"
cp -r "$root/src/pyjoltphysics" "$package/"
cp "$build"/_native*.so "$package/pyjoltphysics/"
runtimes="$(g++ -print-file-name=libasan.so):$(g++ -print-file-name=libubsan.so)"
cd "$root"
LD_PRELOAD="$runtimes" \
ASAN_OPTIONS="detect_leaks=0:abort_on_error=1:symbolize=1" \
UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" \
PYTHONPATH="$package" \
    "$python" -m pytest -p no:cacheprovider -o addopts="--import-mode=importlib" -q tests "$@"
