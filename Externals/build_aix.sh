#!/bin/bash

#
#  Copyright (c) 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

lib_name=aix
lib_version=main
lib_url=git@github.com:godrays/aix.git
logical_core_count=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu)

pushd .

git clone --recurse-submodules -b $lib_version $lib_url ./$lib_name/$lib_version
mkdir -p $lib_name/$lib_version
cd $lib_name/$lib_version

cd Externals
./build_all.sh
cd ..

rm -rf build
rm -rf installed
mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../installed -DAIX_BUILD_STATIC=ON
cmake --build . --target install -- -j $logical_core_count

cd ..
rm -rf .git .gitattributes .github .gitignore build

popd
