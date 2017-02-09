#pragma once
#include <string>
#include "RaytracerConfig.h"

//utility functions to evaluate paths using cmake defined variables

inline std::string get_obj_path(std::string&& p_obj_name) {
	std::string s = MESH_DIR;
	s += '/' + p_obj_name;
	return s;
}

//textures are stored with obj files
inline std::string get_texture_path(std::string&& p_tex_name) {
	std::string s = MESH_DIR;
	s += '/' + p_tex_name;
	return s;
}

inline std::string get_image_path(std::string&& p_image_name) {
	std::string s = IMAGE_OUTPUT_DIR;
	s += '/' + p_image_name;
	return s;
}

inline std::string get_log_path(std::string&& p_log_name) {
	std::string s = LOGS_DIR;
	s += '/' + p_log_name;
	return s;
}