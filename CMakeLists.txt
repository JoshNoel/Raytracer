#Dir: Root
cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)
project (Raytracer VERSION 1.0)

SET(CXX_STANDARD 11)
SET(CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")


#find_package(CUDA QUIET)

if(CUDA_FOUND)
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--gpu-architecture=sm_50;-std=c++11")
	set(NVIDIA_SAMPLES_DIR "" CACHE PATH "Path to nvidia samples include directory for use in utility functions")
	INCLUDE_DIRECTORIES("${NVIDIA_SAMPLES_DIR}/common/inc")
	if(NVIDIA_SAMPLES_DIR STREQUAL "")
		message(SEND_ERROR "NVIDIA_SAMPLES_DIR must be set.")
	endif()
	
	add_definitions(-DUSE_CUDA)
	add_definitions(-DGLM_FORCE_CUDA)
endif()

SET(EXECUTABLE_OUTPUT_PATH bin)

INCLUDE_DIRECTORIES(src)


	
set(GLM_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/glm" CACHE PATH "Path to glm include directory")
INCLUDE_DIRECTORIES(${GLM_INCLUDE_DIR})

set(LODEPNG_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/lodepng" CACHE PATH "Path to lodepng")
INCLUDE_DIRECTORIES(${LODEPNG_DIR})

set(MESH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs/models" CACHE PATH "Path to folder containing mesh files" )
set(IMAGE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs/examples" CACHE PATH "Path to folder containing image files" )
set(LOGS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/docs/logs" CACHE PATH "Path to folder containing log files" )

add_definitions(-DMESH_DIR=${MESH_DIR})
add_definitions(-DIMAGE_OUTPUT_DIR=${IMAGE_OUTPUT_DIR})
add_definitions(-DLOGS_DIR=${LOGS_DIR})
add_definitions(-DGLM_ENABLE_EXPERIMENTAL)

CONFIGURE_FILE( src/RaytracerConfig.h.in config/RaytracerConfig.h)
INCLUDE_DIRECTORIES("${PROJECT_BINARY_DIR}/config")

set(SOURCE_FILES 
	src/BoundingBox.cpp
	src/Camera.cpp			
	src/GeometryObj.cpp		
	src/Image.cpp			
	src/Intersection.cpp
	src/Light.cpp			
	src/Logger.cpp			
	src/Main.cpp
	src/Material.cpp		
	src/Node.cpp			
	src/Plane.cpp			
	src/Ray.cpp				
	src/Renderer.cpp		
	src/Scene.cpp			
	src/Shape.cpp			
	src/Sphere.cpp			
	src/TriObject.cpp		
	src/Triangle.cpp		
	src/Texture.cpp			
	src/ThreadPool.cpp		
	src/WorkQueue.cpp		
	src/Core.cpp			
	src/CudaLoader.cpp)	
#need to set .cpp files to cuda source files so they are compiled with nvcc rather than host compiler
set_source_files_properties(${SOURCE_FILES} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

if(CUDA_FOUND)
	CUDA_ADD_EXECUTABLE(Raytracer
		${SOURCE_FILES}
		${LODEPNG_DIR}/lodepng.cpp
		${PROJECT_BINARY_DIR}/config/RaytracerConfig.h
		src/BoundingBox.h
		src/Camera.h
		src/GeometryObj.h
		src/Image.h
		src/Intersection.h
		src/Light.h
		src/Logger.h
		src/helper/array.h
		src/helper/resource_helper.h
		src/Material.h
		src/Node.h
		src/Plane.h
		src/Ray.h
		src/Renderer.h
		src/Scene.h
		src/Shape.h
		src/Sphere.h
		src/TriObject.h
		src/Triangle.h
		src/Texture.h 
		src/ThreadPool.h
		src/WorkQueue.h
		src/Core.h
		src/CudaLoader.h)
else()
	ADD_EXECUTABLE(Raytracer
		${SOURCE_FILES}
		${LODEPNG_DIR}/lodepng.cpp
		${PROJECT_BINARY_DIR}/config/RaytracerConfig.h
		src/BoundingBox.h
		src/Camera.h
		src/GeometryObj.h
		src/Image.h
		src/Intersection.h
		src/Light.h
		src/Logger.h
		src/helper/array.h
		src/helper/resource_helper.h
		src/Material.h
		src/Node.h
		src/Plane.h
		src/Ray.h
		src/Renderer.h
		src/Scene.h
		src/Shape.h
		src/Sphere.h
		src/TriObject.h
		src/Triangle.h
		src/Texture.h 
		src/ThreadPool.h
		src/WorkQueue.h
		src/Core.h
		src/CudaLoader.h)
endif()