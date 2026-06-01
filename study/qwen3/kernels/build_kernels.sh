#!/bin/bash
# Build the C kernels shared library.
set -e
cd "$(dirname "$0")"
CC="${CC:-gcc}"
FLAGS="-O3 -march=native -ffast-math -fopenmp -shared -fPIC -D_GNU_SOURCE -Wall -Wextra"
echo "$CC $FLAGS kernels.c -o libqwen3_kernels.so -lm"
$CC $FLAGS kernels.c -o libqwen3_kernels.so -lm
ls -lh libqwen3_kernels.so
