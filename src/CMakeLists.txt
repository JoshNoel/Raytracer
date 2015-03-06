#Dir: Root/src

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

CONFIGURE_FILE(
	RaytracerConfig.h.in RaytracerConfig.h
	)
ADD_EXECUTABLE(Raytracer
	Camera.cpp
	Main.cpp
	Image.cpp
	Light.cpp
	Material.cpp
	Math.cpp
	Object.cpp
	Plane.cpp
	Ray.cpp
	Renderer.cpp
	Sphere.cpp
	TriObject.cpp
	)

	TARGET_LINK_LIBRARIES(Raytracer LodePNG)
