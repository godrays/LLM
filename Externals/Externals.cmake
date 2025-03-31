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
function(add_external_git_project)
    set(options)
    set(oneValueArgs NAME GIT_REPOSITORY GIT_TAG GIT_SHALLOW EXTERNALS_BIN_DIR BUILD_TYPE)
    set(multiValueArgs CMAKE_ARGS)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    message(STATUS "Configuring External Project: ${ARG_NAME}")
    set(lib_dir "${ARG_EXTERNALS_BIN_DIR}/${ARG_NAME}")

    # By default, GIT_SHALLOW is ON.
    if ("${ARG_GIT_SHALLOW}" STREQUAL "")
        set(ARG_GIT_SHALLOW ON)
    endif()

    ExternalProject_Add(
            ${ARG_NAME}
            GIT_REPOSITORY  ${ARG_GIT_REPOSITORY}
            GIT_TAG         ${ARG_GIT_TAG}
            GIT_SHALLOW     ${ARG_GIT_SHALLOW}
            PREFIX          "${lib_dir}/prefix"
            SOURCE_DIR      "${lib_dir}/src"
            STAMP_DIR       "${lib_dir}/stamp"
            BINARY_DIR      "${lib_dir}/build"
            INSTALL_DIR     "${lib_dir}/install"
            DOWNLOAD_DIR    "${lib_dir}/download"
            LOG_DIR         "${lib_dir}/log"
            CMAKE_ARGS      -DCMAKE_BUILD_TYPE=${ARG_BUILD_TYPE}
                            -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                            ${ARG_CMAKE_ARGS}
            LOG_DOWNLOAD ON
            LOG_CONFIGURE ON
            LOG_BUILD ON
            LOG_INSTALL ON
            LOG_UPDATE ON
            LOG_PATCH ON
            LOG_TEST ON
            LOG_MERGED_STDOUTERR ON
            LOG_OUTPUT_ON_FAILURE ON
            GIT_SUBMODULES_RECURSE ON
            GIT_PROGRESS OFF
            BUILD_ALWAYS OFF
            UPDATE_COMMAND ""
    )

    # Make include and lib folders available to prevent linker warnings.
    file(MAKE_DIRECTORY "${lib_dir}/install/include" "${lib_dir}/install/lib")

    include_directories(${lib_dir}/install/include)
    link_directories(${lib_dir}/install/lib)
endfunction()

# ---------------------------------------------------------------------------------
# COMMON SETTINGS
# ---------------------------------------------------------------------------------

# Externals build and install folder.
set(EXTERNALS_BINARY_DIR "${CMAKE_BINARY_DIR}/Externals")

# Common cmake project settings for the external projects.
set(EXTERNAL_COMMON_CMAKE_ARGS
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}
        -DCMAKE_POSITION_INDEPENDENT_CODE=${CMAKE_POSITION_INDEPENDENT_CODE}
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=${CMAKE_BUILD_WITH_INSTALL_RPATH}
        -DCMAKE_INSTALL_RPATH=${CMAKE_INSTALL_RPATH}
)

# ---------------------------------------------------------------------------------
# CLEAN EXTERNALS TARGET (Cleans only the external projects)
# ---------------------------------------------------------------------------------

add_custom_target(clean_externals
        COMMAND ${CMAKE_COMMAND} -E remove_directory ${EXTERNALS_BINARY_DIR}
        COMMENT "Cleaning external projects."
)

# ---------------------------------------------------------------------------------
# AIX CPP
# ---------------------------------------------------------------------------------
add_external_git_project(
        NAME                aix_cpp
        GIT_REPOSITORY      https://github.com/godrays/AIX.git
        GIT_TAG             ${EXTERNAL_AIX_VERSION}
        GIT_SHALLOW         OFF
        CMAKE_ARGS          ${EXTERNAL_COMMON_CMAKE_ARGS}
                            -DAIX_BUILD_EXAMPLES=OFF
                            -DAIX_BUILD_TESTS=OFF
                            -DAIX_BUILD_STATIC=ON
        EXTERNALS_BIN_DIR   ${EXTERNALS_BINARY_DIR}
        BUILD_TYPE          Release
)

# ---------------------------------------------------------------------------------
# DOCOPT CPP
# ---------------------------------------------------------------------------------
add_external_git_project(
        NAME                docopt_cpp
        GIT_REPOSITORY      https://github.com/docopt/docopt.cpp.git
        GIT_TAG             ${EXTERNAL_DOCOPT_VERSION}
        GIT_SHALLOW         OFF
        CMAKE_ARGS          ${EXTERNAL_COMMON_CMAKE_ARGS}
                            -DBUILD_SHARED_LIBS=OFF
        EXTERNALS_BIN_DIR   ${EXTERNALS_BINARY_DIR}
        BUILD_TYPE          Release
)
