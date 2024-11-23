#!/bin/bash
set -e

rm -rf code xla torch_xla mlir stablehlo

git clone --depth 1 https://github.com/openxla/xla.git xla
git clone --depth 1 https://github.com/pytorch/xla.git torch_xla
git clone --depth 1 https://github.com/llvm/llvm-project.git mlir
git clone --depth 1 https://github.com/openxla/stablehlo.git

mkdir -p code/xla
mkdir -p code/torch_xla
mkdir -p code/mlir
mkdir -p code/stablehlo

cp -arf xla/xla/ code/xla
cp -arf torch_xla/torch_xla code/torch_xla
cp -arf mlir/mlir/ code/mlir
cp -arf stablehlo/stablehlo/ code/stablehlo

find code -name tests -type d | xargs -n1 rm -rfv
find code -name test -type d | xargs -n1 rm -rfv
find code -name "*_test.cc" -type f | xargs -n1 rm -rfv
