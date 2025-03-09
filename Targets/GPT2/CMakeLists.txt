#
#  Copyright © 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

set(TARGET_NAME GPT2)

add_executable(${TARGET_NAME}
        Main.cpp
)

add_dependencies(${TARGET_NAME} aix_cpp docopt_cpp)

target_link_libraries(${TARGET_NAME} PRIVATE
                      AIXLib
                      docopt
)

if (APPLE AND CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    target_link_libraries(${TARGET_NAME} PRIVATE
                          "-framework Foundation"
                          "-framework Metal"
    )
endif()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION .
)

# Install the entire Resource directory and its contents to the binary directory.
install(DIRECTORY ${CMAKE_SOURCE_DIR}/Resources/
        DESTINATION Resources
        FILES_MATCHING PATTERN "*"
)
