#!/usr/bin/bash
mkdir -p build-linux
cd build-linux
cmake -DNCNN_VULKAN=OFF -DNCNN_BUILD_EXAMPLES=ON -DNCNN_BUILD_TESTS=ON ..
make -j$(nproc)