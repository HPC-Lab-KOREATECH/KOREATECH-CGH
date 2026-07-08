// ======================================================================== //
// Copyright 2018-2019 Ingo Wald
//
// Licensed under the Apache License, Version 2.0
// ======================================================================== //

#include "RUCGH.h"
#include "RTCGHParser.h"

#include <omp.h>

namespace {
void ensureDirectory(const std::filesystem::path& path)
{
	if (!path.empty() && !std::filesystem::exists(path)) {
		std::filesystem::create_directories(path);
	}
}
}

int main(int argc, char** argv) {
	omp_set_nested(1);
	argparse::ArgumentParser program("RUCGH");

	ucgh::parser::GenerationParser generation_command;
	program.add_subparser(*(generation_command.command_));

	if (argc <= 1) {
		std::cout << program;
	}

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::runtime_error& err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		std::exit(1);
	}

	if (program.is_used("-h")) {
		std::cout << program;
		return 0;
	}

	if (!program.is_subcommand_used(generation_command.command_name_)) {
		std::cerr << "Only the 'gen --rgbd_only' command is supported in this release." << std::endl;
		return 1;
	}

	auto rucgh_param = generation_command.makeParams();
	if (!rucgh_param.is_rgbdonly) {
		std::cerr << "This release only supports RGB+D export. Re-run with --rgbd_only." << std::endl;
		return 1;
	}

	if (rucgh_param.num_devices > 1 && rucgh_param.num_devices < std::numeric_limits<short>::max()) {
		std::cout << std::format("Using {} GPUs", rucgh_param.num_devices) << std::endl;
	}
	else {
		std::cout << "device id : " << rucgh_param.device_number << std::endl;
		cudaSetDevice(rucgh_param.device_number);
	}

	ucgh::RGBDRenderParameter param(rucgh_param.width, rucgh_param.height,
		rucgh_param.wavelength[0], rucgh_param.wavelength[1], rucgh_param.wavelength[2], rucgh_param.pixel_pitch, rucgh_param.pixel_pitch);
	int ray_width = rucgh_param.width;
	int ray_height = rucgh_param.height;

	std::vector<ucgh::RUCGH<float>*> cgh_handlers(rucgh_param.num_devices * 2);
	int device_id = 0;
	for (auto& cgh_handler : cgh_handlers)
	{
		cudaSetDevice((device_id++) / 2);
		cgh_handler = new ucgh::RUCGH<float>(param);
	}

	std::filesystem::path depth_path = rucgh_param.depth_output_path;
	std::filesystem::path rgb_path = rucgh_param.rgb_output_path;
	std::filesystem::path csv_path = rucgh_param.csv_path;

	ensureDirectory(depth_path);
	ensureDirectory(rgb_path);
	ensureDirectory(csv_path);

	omp_lock_t* gpu_locks = new omp_lock_t[cgh_handlers.size() / 2];
	for (size_t i = 0; i < cgh_handlers.size() / 2; i++)
	{
		omp_init_lock(gpu_locks + i);
	}
#pragma omp parallel for schedule(dynamic) num_threads(cgh_handlers.size())
	for (int64_t xy = rucgh_param.begin_index; xy < rucgh_param.end_index; xy++) {
		int device_id = omp_get_thread_num() / 2;
		cudaSetDevice(device_id);
		auto& cgh_handler = *cgh_handlers[omp_get_thread_num()];
		auto csv_i_path = (csv_path / std::format("{}.csv", xy));

		if (cgh_handlers.size() <= 2 && rucgh_param.device_number != 0) {
			device_id = rucgh_param.device_number;
		}

		if (!rucgh_param.is_loadmode) {
			if (std::filesystem::exists(csv_i_path)) {
				std::cerr << csv_i_path.string() + " already exists" << std::endl;
				continue;
			}

			std::string record{};
			{
				auto t1 = CUR_TIME;
				cgh_handler.loadMultipleObjects(rucgh_param.obj_path, { rucgh_param.object_mindepth, rucgh_param.object_maxdepth }, rucgh_param.num_objects, record);
				auto t2 = CUR_TIME;
				std::cout << "loading: " << DUR_MICRO(t1, t2).count() / 1000 << "ms" << std::endl;
			}
			if (!rucgh_param.floor_texture_path.empty()) {
				cgh_handler.createPlane(rucgh_param.object_maxdepth, rucgh_param.floor_texture_path, record);
			}
			omp_set_lock(&gpu_locks[device_id]);

			cgh_handler.setRayHandler(gdt::vec2i(ray_width, ray_height), device_id, rucgh_param.ptx_path);
			std::ofstream record_stream(csv_i_path.string());
			record_stream << record << std::flush;
			record_stream.close();
		}
		else {
			{
				auto t1 = CUR_TIME;
				cgh_handler.loadMultipleObjects(rucgh_param.obj_path, csv_i_path);
				auto t2 = CUR_TIME;
				std::cout << "loading: " << DUR_MICRO(t1, t2).count() / 1000 << "ms" << std::endl;
			}
			omp_set_lock(&gpu_locks[device_id]);
			if (!rucgh_param.floor_texture_path.empty()) {
				cgh_handler.loadPlane(rucgh_param.object_maxdepth, rucgh_param.floor_texture_path, csv_i_path);
			}
			cgh_handler.setRayHandler(gdt::vec2i(ray_width, ray_height), device_id, rucgh_param.ptx_path);
		}

		auto& sample = cgh_handler.ray_handler_;
		sample->switchCameraType(ucgh::CameraType::kOrthographic);
		gdt::vec3f camera_position(0, 0, 0);
		gdt::vec3f camera_vertical(0, 1.f, 0);
		gdt::vec3f camera_horizontal(1.f, 0, 0);
		gdt::vec3f camera_direction(0, 0, 1.f / (2.0 * param.getCutoffX()));

		if (rucgh_param.object_maxdepth < 0) {
			camera_direction[2] = -1.f / (2.0 * param.getCutoffX());
		}
		sample->setCamera(camera_position, camera_vertical, camera_horizontal, camera_direction);
		sample->render();

		std::vector<float> pixels(static_cast<size_t>(ray_width) * ray_height * 3);
		std::vector<float> pixels_hwc(static_cast<size_t>(ray_width) * ray_height * 3);
		std::vector<float> depth(static_cast<size_t>(ray_width) * ray_height);
		std::vector<float> z_coord(static_cast<size_t>(ray_width) * ray_height);

		cudaMemcpy(pixels.data(), thrust::raw_pointer_cast(cgh_handler.ray_handler_->framebuffer_), sizeof(decltype(pixels[0])) * pixels.size(), cudaMemcpyDeviceToHost);
		cudaMemcpy(depth.data(), thrust::raw_pointer_cast(cgh_handler.ray_handler_->depthbuffer_), sizeof(decltype(depth[0])) * depth.size(), cudaMemcpyDeviceToHost);
		cudaMemcpy(z_coord.data(), thrust::raw_pointer_cast(cgh_handler.ray_handler_->z_buffer_), sizeof(decltype(depth[0])) * depth.size(), cudaMemcpyDeviceToHost);

#pragma omp parallel for
		for (int64_t i = 0; i < ray_width * ray_height; i++)
		{
			pixels_hwc[i * 3] = pixels[2 * ray_width * ray_height + i];
			pixels_hwc[i * 3 + 1] = pixels[ray_width * ray_height + i];
			pixels_hwc[i * 3 + 2] = pixels[i];
		}

		auto pixel_mat = cv::Mat(ray_height, ray_width, CV_32FC3, pixels_hwc.data());
		auto z_mat = cv::Mat(ray_height, ray_width, CV_32FC1, z_coord.data());
		const std::vector<int> exr_params = { cv::IMWRITE_EXR_TYPE, cv::IMWRITE_EXR_TYPE_FLOAT };
		cv::imwrite((depth_path / std::format("{}.exr", xy)).string(), z_mat, exr_params);
		cv::imwrite((rgb_path / std::format("{}.exr", xy)).string(), pixel_mat, exr_params);

		cv::Mat z_mat_uint;
		cv::normalize(z_mat, z_mat_uint, 1, 0, cv::NORM_MINMAX);
		z_mat_uint.convertTo(z_mat_uint, CV_8UC1, 255);
		if (!depth_path.empty())
			cv::imwrite((depth_path / std::format("{}.png", xy)).string(), z_mat_uint);
		pixel_mat.convertTo(pixel_mat, CV_8UC3, 255);
		if (!rgb_path.empty())
			cv::imwrite((rgb_path / std::format("{}.png", xy)).string(), pixel_mat);

		cgh_handler.ray_handler_->clearAll();
		omp_unset_lock(&gpu_locks[device_id]);
	}

	std::cout << '\a';
	return 0;
}
