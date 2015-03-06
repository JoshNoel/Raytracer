#Dir: Root
cmake_minimum_required (VERSION 2.6)
project (Raytracer)
SET(Raytracer_VERSION_MAJOR 1)
SET(Raytracer_VERSION_MINOR 0)

ADD_SUBDIRECTORY(src)

ADD_SUBDIRECTORY(lib)

TARGET_LINK_LIBRARIES(Raytracer lib/lodepng)

INSTALL(TARGETS Raytracer DESTINATION bin)
