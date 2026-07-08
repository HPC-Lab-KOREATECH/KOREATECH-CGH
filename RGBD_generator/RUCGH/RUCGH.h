#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <numbers>
#include <numeric>
#include <limits>
#include <random>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <opencv2/opencv.hpp>
#include <gdt/math/vec.h>

#include "LaunchParams.h"
#include "OBJ_Loader.h"
#include "RayHandler.h"

#define CUR_TIME std::chrono::system_clock::now()
#define DUR_MICRO(START, END) std::chrono::duration_cast<std::chrono::microseconds>(END - START)

namespace ucgh {

	template <typename T>
	concept floating_point = std::is_same<float, T>::value || std::is_same<double, T>::value;

	class RGBDRenderParameter {
	public:
		size_t nx_{};
		size_t ny_{};
		std::vector<double> wavelength_;
		double dx_{};
		double dy_{};

		RGBDRenderParameter() { this->wavelength_.resize(3); }
		RGBDRenderParameter(size_t nx, size_t ny, double wavelength_r, double wavelength_g, double wavelength_b, double dx, double dy) {
			this->nx_ = nx;
			this->ny_ = ny;
			this->dx_ = dx;
			this->dy_ = dy;
			this->wavelength_.resize(3);
			this->wavelength_[0] = wavelength_r;
			this->wavelength_[1] = wavelength_g;
			this->wavelength_[2] = wavelength_b;
		}
		double getCutoffX(int i = 0) const {
			return std::tan(std::asin(wavelength_[i] / (2.0 * dx_)));
		}
	};

	template <ucgh::floating_point T>
	class RUCGH
	{
	public:
		RGBDRenderParameter param_;
		RayHandler* ray_handler_;

		void setRayHandler(gdt::vec2i size, int device_id = 0, std::filesystem::path ptx_path = "");
		void loadMultipleObjects(std::filesystem::path const& file_directory, std::vector<double> const& boundary, size_t num_items, std::string& record);
		void loadMultipleObjects(std::filesystem::path const& file_directory, std::filesystem::path const& record_path);
		void createPlane(float max_z, std::filesystem::path plane_texture_dir, std::string& record);
		void loadPlane(float max_z, std::filesystem::path plane_texture_dir, std::filesystem::path record);

		RUCGH(RGBDRenderParameter param) : param_(param) {
			this->ray_handler_ = new RayHandler;
			OPTIX_CHECK(optixInit());
		}
	};
}
