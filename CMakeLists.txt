#Dir: Root
cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)
SET(CXX_STANDARD 11)
SET(CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -pthread")


SET(EXECUTABLE_OUTPUT_PATH bin)

project (Raytracer)
SET(Raytracer_VERSION_MAJOR 1)
SET(Raytracer_VERSION_MINOR 0)

INCLUDE_DIRECTORIES(src)

CONFIGURE_FILE(
	src/RaytracerConfig.h.in 
	src/RaytracerConfig.h
	)
ADD_EXECUTABLE(Raytracer
	lib/lodepng/lodepng.cpp
	src/BoundingBox.cpp		src/BoundingBox.h
	src/Camera.cpp			src/Camera.h
	src/GeometryObj.cpp		src/GeometryObj.h
	src/Image.cpp			src/Image.h
	src/Intersection.cpp	src/Intersection.h
	src/Light.cpp			src/Light.h
	src/Logger.cpp			src/Logger.h
	src/Main.cpp
	src/Material.cpp		src/Material.h
	src/Node.cpp			src/Node.h
	src/Plane.cpp			src/Plane.h
	src/Ray.cpp				src/Ray.h
	src/Renderer.cpp		src/Renderer.h
	src/Scene.cpp			src/Scene.h
	src/Shape.cpp			src/Shape.h
	src/Sphere.cpp			src/Sphere.h
	src/TriObject.cpp		src/TriObject.h
	src/Triangle.cpp		src/Triangle.h
	src/Texture.cpp			src/Texture.h 
	src/ThreadPool.cpp		src/ThreadPool.h
	src/WorkQueue.cpp		src/WorkQueue.h
	src/Core.cpp			src/Core.h)


set(GLM_INCLUDE_DIR lib/glm)
INCLUDE_DIRECTORIES(${GLM_INCLUDE_DIR})

set(LODEPNG_DIR lib/lodepng)
INCLUDE_DIRECTORIES(${LODEPNG_DIR})
