# ======================================================================== #
# Copyright 2018-2024 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

project(DeepComp VERSION 1.0.0 LANGUAGES C CXX CUDA)

find_package(MPI REQUIRED)

if (NOT (DEFINED CMAKE_CUDA_ARCHITECTURES))
  if (NOT CMAKE_VERSION VERSION_LESS "3.17")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  endif()
endif()

add_subdirectory(../owl build_owl EXCLUDE_FROM_ALL)
add_library(deepCompositing STATIC
  deepCompositing.cu
  )
target_link_libraries(deepCompositing
  PUBLIC
  owl::owl
  MPI::MPI_CXX
  )


add_executable(testDeepOffline
  testDeepOffline.cpp
  testRenderer.cu
  )
target_link_libraries(testDeepOffline
  deepCompositing
  stb_image
  )


add_executable(dcBench
  bench.cu
  )
target_link_libraries(dcBench
  deepCompositing
  owl::owl
  stb_image
  )


add_executable(testDeepViewer
  testDeepViewer.cpp
  testRenderer.cu
  )
target_link_libraries(testDeepViewer
  deepCompositing
  owl::owl
  owl_viewer
  stdc++fs # for experimental::filesystem
  )


add_executable(compositeBricks
  compositeBricks.cpp
  )
target_link_libraries(compositeBricks
  owl::owl
  stdc++fs
  stb_image
  )

