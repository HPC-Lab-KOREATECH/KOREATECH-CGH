#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "optix7.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_free.h>
#include <gdt/math/vec.h>
#include "LaunchParams.h"
#include "OBJ_Loader.h"
#include <opencv2/opencv.hpp>
#include <tuple>
#include <format>
#include <filesystem>
#include <map>
#include <random>
#include <chrono>

#define CUR_TIME std::chrono::system_clock::now()
#define DUR_MICRO(START, END) std::chrono::duration_cast<std::chrono::microseconds>(END - START)

#define CUDA_HELP(call)							\
    {									\
      cudaError_t rc = call;                                      \
      if (rc != cudaSuccess) {                                          \
        std::stringstream txt;                                          \
        cudaError_t err =  rc; /*cudaGetLastError();*/                  \
        txt << "CUDA Error " << cudaGetErrorName(err)                   \
            << " (" << cudaGetErrorString(err) << ")";                  \
        throw std::runtime_error(txt.str());                            \
      }                                                                 \
    }


struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	// just a dummy value - later examples will use more interesting
	// data here
	void* data;
};

/*! SBT record for a miss program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	// just a dummy value - later examples will use more interesting
	// data here
	void* data;
};

/*! SBT record for a hitgroup program */
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) HitgroupRecord
{
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
	osc::TriangleMeshSBTData data;
};



namespace ucgh {

	enum class CameraType
	{
		kOrthographic
	};

	struct Mesh {
		std::vector<gdt::vec3f> vertices_;
		std::vector<gdt::vec3f> normals_;
		std::vector<gdt::vec3i> indices_;
		std::vector<gdt::vec2f> tex_coords_;
		std::vector<std::string> mesh_name_;
		void resize(size_t size) {
			vertices_.resize(size);
			normals_.resize(size);
			tex_coords_.resize(size);
		}
	};

	class RayHandler {
	public:
		objl::Loader mesh_loader_;
		std::vector<ucgh::Mesh> mesh_data_;
		std::vector<cudaTextureObject_t> cuda_texture_objects_;
		std::vector<cudaArray_t> texture_arrays_;
		std::vector<cv::Mat> textures_;

		thrust::device_ptr<float> framebuffer_;
		thrust::device_ptr<float> depthbuffer_;
		thrust::device_ptr<float> z_buffer_;

		std::vector<float> host_framebuffer_;
		std::vector<thrust::device_vector<std::array<float, 3>>> vertex_buffers_;
		std::vector<thrust::device_vector<std::array<float, 3>>> normal_buffers_;
		std::vector<thrust::device_vector<std::array<float, 2>>> texcoord_buffers_;
		std::vector<thrust::device_vector<std::array<int, 3>>> index_buffers_{};

		CUcontext cuda_context_;
		cudaStream_t cuda_stream_;

		OptixModule optix_module_;
		OptixDeviceContext optix_context_;
		OptixPipeline optix_pipeline_;
		OptixPipelineCompileOptions optix_pipeline_compile_options_{};
		OptixPipelineLinkOptions optix_pipeline_link_options_{};
		osc::LaunchParams launch_params_host_{};
		thrust::device_ptr<osc::LaunchParams> launch_params_;


		std::vector<OptixProgramGroup> optix_raygens_;
		thrust::device_ptr<RaygenRecord> raygen_records_buffer_;
		std::vector<OptixProgramGroup> optix_misss_;
		thrust::device_ptr<MissRecord> miss_records_buffer_;
		std::vector<OptixProgramGroup> optix_hitgroups_;
		thrust::device_ptr<HitgroupRecord> hitgroup_records_buffer_;
		OptixShaderBindingTable optix_shader_binding_table_{};


		thrust::device_ptr<char> accel_structure_buffer_;

		void loadObj(std::filesystem::path obj_path, std::filesystem::path material_path, std::filesystem::path texture_path, std::array<double, 3> const& scale = { 0,0,0 }, std::array<double, 3> const& rotation_angle = { 0,0,0 }, std::array<double, 3> const& translation = { 0,0,0 });
		void init(int device_id =0, std::filesystem::path ptx_path = "");
		void render();
		void initOptixContext(int deviceID = 0);
		void initOptixModule(std::filesystem::path ptx_path = "");
		void initOptixRaygen(ucgh::CameraType camera_type = ucgh::CameraType::kOrthographic);
		void initOptixMiss();
		void initOptixHitgroup();
		void initOptixPipeline();
		void initOptixSBT();
		void createTextures();
		void clearOptixPipeline();
		void clearOptixContext();
		void clearOptixModule();
		void clearProgramGroups();
		void clearTextures();
		void clearObjects();
		void clearOptixBuffer();

		void setBufferSize(gdt::vec2i size);
		void clearBuffer();
		void clearAll();
		void setCamera(gdt::vec3f position = gdt::vec3f(0, 0, 0), gdt::vec3f vertical = gdt::vec3f(0, 1.f, 0),
			gdt::vec3f horizontal = gdt::vec3f(1.f, 0, 0), gdt::vec3f direction = gdt::vec3f(0, 0, -1.f));
		void switchCameraType(ucgh::CameraType camera_type);

		OptixTraversableHandle buildAccel();
	};
}


