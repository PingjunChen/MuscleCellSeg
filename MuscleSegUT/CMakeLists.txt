CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

PROJECT(MuscleSegUT CXX)
OPTION(BUILD_MUSCLE_SEG_UT "Build MuscleSeg unittest." ON)

IF(BUILD_MUSCLE_SEG_UT)
    FILE(GLOB_RECURSE MUSCLE_SEG_UT_SRC ./*.cpp)
    SOURCE_GROUP("Source Files" FILES ${MUSCLE_SEG_UT_SRC})

    INCLUDE_DIRECTORIES(
        ${MUSCLEMINER_SOURCE_DIR}/include/
        ${DEPENDENCE_PATH}/gtest/include
        ${DEPENDENCE_PATH}/OpenCV-3.0.0/include
    )

    LINK_DIRECTORIES(
        ${PROJECT_BINARY_DIR}/../bin/Debug
        ${DEPENDENCE_PATH}/gtest/lib
        ${DEPENDENCE_PATH}/OpenCV-3.0.0/lib
    )

    # ADD_DEPENDENCIES(muscle_seg_ut muscle_seg)
    ADD_EXECUTABLE(muscle_seg_ut ${MUSCLE_SEG_UT_SRC})
    TARGET_LINK_LIBRARIES(muscle_seg_ut muscle_seg.lib gtest.lib opencv_world300d.lib)

    SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)
ENDIF(BUILD_MUSCLE_SEG_UT)
