#Dir: Root
cmake_minimum_required (VERSION 3.1.0 FATAL_ERROR)
SET(CXX_STANDARD 11)
SET(CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -pthread")

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
	src/Texture.cpp)


set(GLM_INCLUDE_DIR lib/glm)
INCLUDE_DIRECTORIES(${GLM_INCLUDE_DIR})

set(LODEPNG_DIR lib/lodepng)
INCLUDE_DIRECTORIES(${LODEPNG_DIR})
