#
#  Copyright © 2024-Present, Arkin Terli. All rights reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of Arkin Terli.
#  The intellectual and technical concepts contained herein are proprietary to Arkin Terli
#  and may be covered by U.S. and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained from Arkin Terli.

set(TARGET_NAME GPT2)
set(RESOURCES_DIR Resources/GPT2)

add_executable(${TARGET_NAME}
        main.cpp
)

# Record files names in RESOURCES_DIR.
file(GLOB_RECURSE RESOURCE_FILES
    "${CMAKE_SOURCE_DIR}/${RESOURCES_DIR}/*"
)

# Make the resource directories.
add_custom_command(TARGET ${TARGET_NAME} PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/${RESOURCES_DIR})

# Copy each file into the binary folder if they are not exist or different.
foreach(RESOURCE_FILE ${RESOURCE_FILES})
    # Strip the leading path to make it relative to Resources/GPT2.
    file(RELATIVE_PATH RELATIVE_FILE_PATH ${CMAKE_SOURCE_DIR}/${RESOURCES_DIR} ${RESOURCE_FILE})

    # Custom target to copy resources always if the resource is different.
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${RESOURCE_FILE}
                       ${CMAKE_CURRENT_BINARY_DIR}/${RESOURCES_DIR}/
    )
endforeach()

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

# Install the entire Resource/GPT2 directory and its contents to the binary directory.
install(DIRECTORY ${CMAKE_SOURCE_DIR}/${RESOURCES_DIR}/
        DESTINATION ${RESOURCES_DIR}
        FILES_MATCHING PATTERN "*"
)
