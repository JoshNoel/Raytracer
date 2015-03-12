#Dir: Root
cmake_minimum_required (VERSION 2.6)
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
	src/Camera.cpp
	src/Main.cpp
	src/Image.cpp
	src/Light.cpp
	src/Material.cpp
	src/Math.cpp
	src/Object.cpp	
	src/Plane.cpp
	src/Ray.cpp
	src/Renderer.cpp
	src/Sphere.cpp
	src/TriObject.cpp
	src/BoundingBox.cpp
	)


set(GLM_INCLUDE_DIR lib/glm)
INCLUDE_DIRECTORIES(${GLM_INCLUDE_DIR})

set(LODEPNG_DIR lib/lodepng)
INCLUDE_DIRECTORIES(${LODEPNG_DIR})