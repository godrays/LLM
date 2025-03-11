#
#  Copyright © 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

include(ExternalProject)

# ---------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------------

# Builds and installs external git projects.
function(add_external_git_project lib_name git_repository git_tag cmake_project_args external_bin_dir build_type)
    message(STATUS "Configuring External Project: ${lib_name}")
    ExternalProject_Add(
            ${lib_name}
            GIT_REPOSITORY ${git_repository}
            GIT_TAG        ${git_tag}
            PREFIX        "${external_bin_dir}/${lib_name}/prefix"
            SOURCE_DIR    "${external_bin_dir}/${lib_name}/src"
            STAMP_DIR     "${external_bin_dir}/${lib_name}/stamp"
            BINARY_DIR    "${external_bin_dir}/${lib_name}/build"
            INSTALL_DIR   "${external_bin_dir}/${lib_name}/install"
            DOWNLOAD_DIR  "${external_bin_dir}/${lib_name}/download"
            LOG_DIR       "${external_bin_dir}/${lib_name}/log"
            CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${build_type}
            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
            ${cmake_project_args}       # Project Build Options
            LOG_CONFIGURE ON
            LOG_BUILD ON
            LOG_INSTALL ON
            LOG_UPDATE ON
            LOG_PATCH ON
            LOG_TEST ON
            LOG_MERGED_STDOUTERR ON
            LOG_OUTPUT_ON_FAILURE ON
            BUILD_ALWAYS ON
    )
    set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_CLEAN_FILES "${external_bin_dir}/${lib_name}")
    include_directories(${external_bin_dir}/${lib_name}/install/include)
    link_directories(${external_bin_dir}/${lib_name}/install/lib)
endfunction()

# ---------------------------------------------------------------------------------
# COMMON SETTINGS
# ---------------------------------------------------------------------------------

# Common cmake project settings for the external projects.
set(EXTERNAL_COMMON_CMAKE_ARGS
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
        -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}
        -DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_POSITION_INDEPENDENT_CODE=${CMAKE_POSITION_INDEPENDENT_CODE}
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=${CMAKE_BUILD_WITH_INSTALL_RPATH}
        -DCMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}
)

# Externals build and install folder.
set(EXTERNALS_BINARY_DIR "${CMAKE_BINARY_DIR}/Externals")

# ---------------------------------------------------------------------------------
# AIX CPP
# ---------------------------------------------------------------------------------
set(EXTERNAL_AIX_CMAKE_ARGS
        ${EXTERNAL_COMMON_CMAKE_ARGS}
        # Project specific cmake args
        -DAIX_BUILD_EXAMPLES=OFF
        -DAIX_BUILD_TESTS=OFF
        -DAIX_BUILD_STATIC=ON
)

add_external_git_project(
        "aix_cpp"
        "https://github.com/godrays/AIX.git"
        "${EXTERNAL_AIX_VERSION}"
        "${EXTERNAL_AIX_CMAKE_ARGS}"
        "${EXTERNALS_BINARY_DIR}"
        "Release"
)

# ---------------------------------------------------------------------------------
# DOCOPT CPP
# ---------------------------------------------------------------------------------
set(EXTERNAL_DOCOPT_CMAKE_ARGS
        ${EXTERNAL_COMMON_CMAKE_ARGS}
        # Project specific cmake args
        -DBUILD_SHARED_LIBS=OFF
)

add_external_git_project(
        "docopt_cpp"
        "https://github.com/docopt/docopt.cpp.git"
        "${EXTERNAL_DOCOPT_VERSION}"
        "${EXTERNAL_DOCOPT_CMAKE_ARGS}"
        "${EXTERNALS_BINARY_DIR}"
        "Release"
)
