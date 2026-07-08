#include "RayHandler.h"

#include <optix_function_table_definition.h>


inline std::array<double, 9> makeRotationMatrix(double roll, double pitch, double yaw) {
	double cx = cos(roll);
	double sx = sin(roll);
	double cy = cos(pitch);
	double sy = sin(pitch);
	double cz = cos(yaw);
	double sz = sin(yaw);

	std::array<double, 9> R = {
		cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
		sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
		-sy,     cy * sx,                cy * cx
	};

	return R;
}


gdt::vec3f rotatePoint(std::array<double, 9> rotation_matrix, gdt::vec3f vertex) {
	double x = vertex.x * rotation_matrix[0] + vertex.y * rotation_matrix[1] + vertex.z * rotation_matrix[2];
	double y = vertex.x * rotation_matrix[3] + vertex.y * rotation_matrix[4] + vertex.z * rotation_matrix[5];
	double z = vertex.x * rotation_matrix[6] + vertex.y * rotation_matrix[7] + vertex.z * rotation_matrix[8];

	return gdt::vec3f(x, y, z);
}

gdt::vec3f rotatePoint(std::array<double, 9> rotation_matrix, objl::Vertex& vertex, std::array<double, 3> scale = { 1,1,1 }, std::array<double, 3> translation = { 0,0,0 }) {
	double x = (vertex.Position.X * rotation_matrix[0] + vertex.Position.Y * rotation_matrix[1] + vertex.Position.Z * rotation_matrix[2]) * scale[0] + translation[0];
	double y = (vertex.Position.X * rotation_matrix[3] + vertex.Position.Y * rotation_matrix[4] + vertex.Position.Z * rotation_matrix[5]) * scale[1] + translation[1];
	double z = (vertex.Position.X * rotation_matrix[6] + vertex.Position.Y * rotation_matrix[7] + vertex.Position.Z * rotation_matrix[8]) * scale[2] + translation[2];

	double n_x = (vertex.Normal.X * rotation_matrix[0] + vertex.Normal.Y * rotation_matrix[1] + vertex.Normal.Z * rotation_matrix[2]);
	double n_y = (vertex.Normal.X * rotation_matrix[3] + vertex.Normal.Y * rotation_matrix[4] + vertex.Normal.Z * rotation_matrix[5]);
	double n_z = (vertex.Normal.X * rotation_matrix[6] + vertex.Normal.Y * rotation_matrix[7] + vertex.Normal.Z * rotation_matrix[8]);


	vertex.Position.X = x;
	vertex.Position.Y = y;
	vertex.Position.Z = z;

	vertex.Normal.X = n_x;
	vertex.Normal.Y = n_y;
	vertex.Normal.Z = n_z;
	return gdt::vec3f(x, y, z);
}

gdt::vec3f rotatePoint(std::array<double, 9> rotation_matrix, objl::Vertex& vertex, gdt::vec3f scale = { 1,1,1 }, gdt::vec3f translation = { 0,0,0 }) {
	double x = (vertex.Position.X * rotation_matrix[0] + vertex.Position.Y * rotation_matrix[1] + vertex.Position.Z * rotation_matrix[2]) * scale[0] + translation[0];
	double y = (vertex.Position.X * rotation_matrix[3] + vertex.Position.Y * rotation_matrix[4] + vertex.Position.Z * rotation_matrix[5]) * scale[1] + translation[1];
	double z = (vertex.Position.X * rotation_matrix[6] + vertex.Position.Y * rotation_matrix[7] + vertex.Position.Z * rotation_matrix[8]) * scale[2] + translation[2];

	vertex.Position.X = x;
	vertex.Position.Y = y;
	vertex.Position.Z = z;
	return gdt::vec3f(x, y, z);
}

static void context_log_cb(unsigned int level,
	const char* tag,
	const char* message,
	void*)
{
	fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

void ucgh::RayHandler::loadObj(std::filesystem::path obj_path, std::filesystem::path material_path, std::filesystem::path texture_path, std::array<double, 3> const& scale, std::array<double, 3> const& rotation_angle, std::array<double, 3> const& translation)
{
	this->mesh_loader_.LoadFile(obj_path.string());

	auto x_minmax = std::minmax_element(this->mesh_loader_.LoadedVertices.begin(), this->mesh_loader_.LoadedVertices.end(),
		[](auto& vertex1, auto& vertex2) {return std::less<>{}(vertex1.Position.X, vertex2.Position.X); });
	auto y_minmax = std::minmax_element(this->mesh_loader_.LoadedVertices.begin(), this->mesh_loader_.LoadedVertices.end(),
		[](auto& vertex1, auto& vertex2) {return std::less<>{}(vertex1.Position.Y, vertex2.Position.Y); });
	auto z_minmax = std::minmax_element(this->mesh_loader_.LoadedVertices.begin(), this->mesh_loader_.LoadedVertices.end(),
		[](auto& vertex1, auto& vertex2) {return std::less<>{}(vertex1.Position.Z, vertex2.Position.Z); });

	float x_mid_point = (x_minmax.second[0].Position.X + x_minmax.first[0].Position.X) / 2.0;
	float y_mid_point = (y_minmax.second[0].Position.Y + y_minmax.first[0].Position.Y) / 2.0;
	float z_mid_point = (z_minmax.second[0].Position.Z + z_minmax.first[0].Position.Z) / 2.0;
	//std::cout << std::format("Center point : {}, {}, {}", x_mid_point, y_mid_point, z_mid_point) << std::endl;
	float bounding_size_x = x_minmax.second[0].Position.X - x_minmax.first[0].Position.X;
	float bounding_size_y = y_minmax.second[0].Position.Y - y_minmax.first[0].Position.Y;

	auto scaling_ratio = std::max(bounding_size_x, bounding_size_y);


	auto rotation_matrix = makeRotationMatrix(rotation_angle[0], rotation_angle[1], rotation_angle[2]);

	for (auto& mesh : this->mesh_loader_.LoadedMeshes) {
#pragma omp parallel for 
		for (int64_t i = 0; i < mesh.Vertices.size(); i++)
		{
			auto &vertex = mesh.Vertices[i];
			vertex.Position.X = (vertex.Position.X - x_mid_point) / scaling_ratio;
			vertex.Position.Y = (vertex.Position.Y - y_mid_point) / scaling_ratio;
			vertex.Position.Z = (vertex.Position.Z - z_mid_point) / scaling_ratio;
			rotatePoint(rotation_matrix, vertex, scale, translation);
		}
	}

	for (size_t i = 0; i < this->mesh_loader_.LoadedMeshes.size(); i++)
	{
		auto& loaded_mesh = mesh_loader_.LoadedMeshes[i];
		ucgh::Mesh mesh;
		mesh.resize(loaded_mesh.Vertices.size());
		mesh.indices_.resize(loaded_mesh.Indices.size() / 3);
		std::transform(loaded_mesh.Vertices.begin(), loaded_mesh.Vertices.end(), mesh.vertices_.begin(), [](objl::Vertex ele)->gdt::vec3f { gdt::vec3f output; output.x = ele.Position.X; output.y = ele.Position.Y; output.z = ele.Position.Z; return output; });
		std::transform(loaded_mesh.Vertices.begin(), loaded_mesh.Vertices.end(), mesh.normals_.begin(), [](objl::Vertex ele)->gdt::vec3f { gdt::vec3f output; output.x = ele.Normal.X; output.y = ele.Normal.Y; output.z = ele.Normal.Z; return output; });
		std::transform(loaded_mesh.Vertices.begin(), loaded_mesh.Vertices.end(), mesh.tex_coords_.begin(), [](objl::Vertex ele)->gdt::vec2f { gdt::vec2f output; output.x = ele.TextureCoordinate.X; output.y = ele.TextureCoordinate.Y; return output; });

		std::memcpy(mesh.indices_.data(), loaded_mesh.Indices.data(), sizeof(int) * loaded_mesh.Indices.size());

		this->mesh_data_.push_back(mesh);
		std::string texture_name = loaded_mesh.MeshMaterial.map_Kd;
		if (texture_path.empty()) {
			throw std::runtime_error("Texture file not found for object: " + obj_path.string());
		}

		auto texture = cv::imread(texture_path.string(), cv::IMREAD_COLOR);
		if (texture.empty()) {
			throw std::runtime_error("Could not load texture file: " + texture_path.string());
		}
		cv::cvtColor(texture, texture, cv::COLOR_BGR2RGBA);
		this->textures_.push_back(texture);

	}
}


void ucgh::RayHandler::init(int device_id, std::filesystem::path ptx_path)
{
	objl::Loader mesh_loader_;

	//init optix
	createTextures();
	//init raygen program
	this->initOptixContext(device_id);
	this->initOptixModule(ptx_path);
	this->initOptixRaygen(ucgh::CameraType::kOrthographic);
	this->initOptixMiss();
	this->initOptixHitgroup();

	this->launch_params_host_.traversable = buildAccel();
	this->initOptixPipeline();

	this->initOptixSBT();


	this->launch_params_ = thrust::device_malloc<osc::LaunchParams>(1);

	std::cout << "init done" << std::endl;
}

void ucgh::RayHandler::render()
{
	if (launch_params_host_.frame.size[0] == 0 || launch_params_host_.frame.size[1] == 0) {
		std::cerr << "launch parameter error" << std::endl;
		return;
	}

	CUDA_HELP(cudaMemcpyAsync(thrust::raw_pointer_cast(this->launch_params_), &this->launch_params_host_, sizeof(osc::LaunchParams), cudaMemcpyHostToDevice, this->cuda_stream_));
	//launch_params_host_.frameID++;
	auto t1 = CUR_TIME;
	OPTIX_CHECK(
		optixLaunch(/*! pipeline we're launching launch: */
			this->optix_pipeline_, this->cuda_stream_,
			/*! parameters and SBT */
			reinterpret_cast<CUdeviceptr>(thrust::raw_pointer_cast(this->launch_params_)),
			sizeof(osc::LaunchParams),
			&this->optix_shader_binding_table_,
			/*! dimensions of the launch: */
			this->launch_params_host_.frame.size[0],
			this->launch_params_host_.frame.size[1],
			1
		));

	CUDA_SYNC_CHECK();
	auto t2 = CUR_TIME;
	std::cout << "rendering: " << DUR_MICRO(t1, t2).count() / 1000 << "ms" << std::endl;

}

void ucgh::RayHandler::initOptixContext(int deviceID)
{
	/*const int deviceID = 0;
	CUDA_HELP(cudaSetDevice(deviceID));*/
	CUDA_HELP(cudaStreamCreate(&this->cuda_stream_));

	cudaDeviceProp device_props{};
	cudaGetDeviceProperties(&device_props, deviceID);
	std::cout << "#osc: running on device: " << device_props.name << std::endl;

	CU_CHECK(cuCtxGetCurrent(&this->cuda_context_));
	OPTIX_CHECK(optixDeviceContextCreate(this->cuda_context_, 0, &this->optix_context_));
	OPTIX_CHECK(optixDeviceContextSetLogCallback(this->optix_context_, context_log_cb, nullptr, 4));
}

void ucgh::RayHandler::initOptixModule(std::filesystem::path ptx_path)
{
	OptixModuleCompileOptions optix_module_compile_option{};
	optix_module_compile_option.maxRegisterCount = 50;
	optix_module_compile_option.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
	optix_module_compile_option.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

	this->optix_pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
	this->optix_pipeline_compile_options_.usesMotionBlur = false;
	this->optix_pipeline_compile_options_.numPayloadValues = 2;
	this->optix_pipeline_compile_options_.numAttributeValues = 2;
	this->optix_pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
	this->optix_pipeline_compile_options_.pipelineLaunchParamsVariableName = "optixLaunchParams";

	if (std::filesystem::is_directory(ptx_path)) {
#ifdef _WIN64
		ptx_path = ptx_path / "deviceProgramOrthogonal.cu.obj";
#else
		ptx_path = ptx_path / "deviceProgramOrthogonal.ptx";
#endif
	}

	if (!std::filesystem::exists(ptx_path)) {
#ifdef _WIN64
		ptx_path = "x64/Release/deviceProgramOrthogonal.cu.obj";
#else
		ptx_path = "RUCGH/CMakeFiles/myptx.dir/deviceProgramOrthogonal.ptx";

#endif // _WIN64
	}

	std::ifstream ptx_file(ptx_path);
	if (!ptx_file.is_open()) {
		throw std::runtime_error("Could not open PTX file: " + ptx_path.string());
	}

	std::stringstream buffer;
	buffer << ptx_file.rdbuf();
	std::string ptx_code = buffer.str();

	char log[2048]{};
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixModuleCreate(this->optix_context_,
		&optix_module_compile_option,
		&this->optix_pipeline_compile_options_,
		ptx_code.c_str(),
		ptx_code.size(),
		log, &sizeof_log,
		&this->optix_module_
	));
	if (sizeof_log > 1) std::cout << log << std::endl;
}

void ucgh::RayHandler::initOptixRaygen(ucgh::CameraType camera_type)
{
	this->optix_raygens_.resize(1);
	OptixProgramGroupOptions option_dummy = {};
	OptixProgramGroupDesc rangen_desc = {};
	rangen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
	rangen_desc.raygen.module = this->optix_module_;
	if (camera_type == ucgh::CameraType::kOrthographic)
	{
		rangen_desc.raygen.entryFunctionName = "__raygen__renderFrameOrthogonal";
	}
	else {
		std::cout << "camera type error" << std::endl;
		exit(1);
	}


	char log[2048];
	size_t sizeof_log = sizeof(log);
	// OptixProgramGroup raypg;
	OPTIX_CHECK(optixProgramGroupCreate(this->optix_context_,
		&rangen_desc,
		1,
		&option_dummy,
		log, &sizeof_log,
		&this->optix_raygens_[0]
	));
	if (sizeof_log > 1) std::cout << log << std::endl;
}

void ucgh::RayHandler::initOptixMiss()
{
	this->optix_misss_.resize(1);
	OptixProgramGroupOptions option_dummy = {};
	OptixProgramGroupDesc miss_desc = {};
	miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
	miss_desc.miss.module = this->optix_module_;
	miss_desc.miss.entryFunctionName = "__miss__radiance";

	char log[2048];
	size_t sizeof_log = sizeof(log);

	OPTIX_CHECK(optixProgramGroupCreate(this->optix_context_,
		&miss_desc,
		1,
		&option_dummy,
		log, &sizeof_log,
		&this->optix_misss_[0]
	));
	if (sizeof_log > 1) std::cout << (log) << std::endl;
}

void ucgh::RayHandler::initOptixHitgroup()
{
	// for this simple example, we set up a single hit group
	this->optix_hitgroups_.resize(1);

	OptixProgramGroupOptions option_dummy = {};
	OptixProgramGroupDesc hit_desc = {};
	hit_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
	hit_desc.hitgroup.moduleCH = this->optix_module_;
	hit_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
	hit_desc.hitgroup.moduleAH = this->optix_module_;
	hit_desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixProgramGroupCreate(this->optix_context_,
		&hit_desc,
		1,
		&option_dummy,
		log, &sizeof_log,
		&this->optix_hitgroups_[0]
	));
	if (sizeof_log > 1) std::cout << (log) << std::endl;
}

void ucgh::RayHandler::initOptixPipeline()
{
	std::vector<OptixProgramGroup> programGroups;
	for (auto pg : this->optix_raygens_)
		programGroups.push_back(pg);
	for (auto pg : this->optix_misss_)
		programGroups.push_back(pg);
	for (auto pg : this->optix_hitgroups_)
		programGroups.push_back(pg);

	char log[2048];
	size_t sizeof_log = sizeof(log);
	OPTIX_CHECK(optixPipelineCreate(this->optix_context_,
		&this->optix_pipeline_compile_options_,
		&this->optix_pipeline_link_options_,
		programGroups.data(),
		(int)programGroups.size(),
		log, &sizeof_log,
		&this->optix_pipeline_
	));
	if (sizeof_log > 1) std::cout << (log) << std::endl;

	OPTIX_CHECK(optixPipelineSetStackSize
	(/* [in] The pipeline to configure the stack size for */
		this->optix_pipeline_,
		/* [in] The direct stack size requirement for direct
		   callables invoked from IS or AH. */
		2 * 1024,
		/* [in] The direct stack size requirement for direct
		   callables invoked from RG, MS, or CH.  */
		2 * 1024,
		/* [in] The continuation stack requirement. */
		2 * 1024,
		/* [in] The maximum depth of a traversable graph
		   passed to trace. */
		1));
	if (sizeof_log > 1) std::cout << (log) << std::endl;
}

void ucgh::RayHandler::initOptixSBT()
{

	this->raygen_records_buffer_ = thrust::device_malloc<RaygenRecord>(this->optix_raygens_.size());
	std::vector<RaygenRecord> raygen_records_host;
	for (int i = 0; i < this->optix_raygens_.size(); i++) {
		RaygenRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(this->optix_raygens_[i], &rec));
		rec.data = nullptr; /* for now ... */
		raygen_records_host.push_back(rec);
	}

	cudaMemcpy(thrust::raw_pointer_cast(this->raygen_records_buffer_), raygen_records_host.data(), raygen_records_host.size() * sizeof(RaygenRecord), cudaMemcpyHostToDevice);
	this->optix_shader_binding_table_.raygenRecord = reinterpret_cast<CUdeviceptr>(thrust::raw_pointer_cast(this->raygen_records_buffer_));

	// ------------------------------------------------------------------
	// build miss records
	// ------------------------------------------------------------------
	this->miss_records_buffer_ = thrust::device_malloc<MissRecord>(this->optix_misss_.size());
	std::vector<MissRecord> miss_records_host;
	for (int i = 0; i < this->optix_misss_.size(); i++) {
		MissRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(this->optix_misss_[i], &rec));
		rec.data = nullptr; /* for now ... */
		miss_records_host.push_back(rec);
	}

	cudaMemcpy(thrust::raw_pointer_cast(this->miss_records_buffer_), miss_records_host.data(), miss_records_host.size() * sizeof(MissRecord), cudaMemcpyHostToDevice);
	this->optix_shader_binding_table_.missRecordBase = reinterpret_cast<CUdeviceptr>(thrust::raw_pointer_cast(this->miss_records_buffer_));
	this->optix_shader_binding_table_.missRecordStrideInBytes = sizeof(MissRecord);
	this->optix_shader_binding_table_.missRecordCount = (int)this->optix_misss_.size();

	// ------------------------------------------------------------------
	// build hitgroup records
	// ------------------------------------------------------------------

	// we don't actually have any objects in this example, but let's
	// create a dummy one so the SBT doesn't have any null pointers
	// (which the sanity checks in compilation would complain about)
	/*int numObjects = 1;
	for (int i = 0; i < numObjects; i++) {
		int objectType = 0;
		HitgroupRecord rec;
		OPTIX_CHECK(optixSbtRecordPackHeader(this->optix_hitgroups_[objectType], &rec));
		rec.objectID = i;
		this->hitgroup_records_buffer_.push_back(rec);
	}

	HitgroupRecord* hitgroup_records_buffer_pointer = thrust::raw_pointer_cast(this->hitgroup_records_buffer_.data());

	this->optix_shader_binding_table_.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_records_buffer_pointer);
	this->optix_shader_binding_table_.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	this->optix_shader_binding_table_.hitgroupRecordCount = (int)this->optix_hitgroups_.size();*/

	int num_objects = this->mesh_data_.size();

	std::vector<HitgroupRecord> hitgroupRecords;
	for (int meshID = 0; meshID < num_objects; meshID++) {
		auto mesh = this->mesh_data_;

		HitgroupRecord rec;
		// all meshes use the same code, so all same hit group
		OPTIX_CHECK(optixSbtRecordPackHeader(this->optix_hitgroups_[0], &rec));
		rec.data.color[0] = 1.0f;
		rec.data.color[1] = 1.0f;
		//rec.data.color[2] = this->mesh_loader_.LoadedMaterials[0].Kd.Z;
		rec.data.color[2] = 1.0f;
		rec.data.hasTexture = false;
		if (!this->mesh_loader_.LoadedMaterials.empty() &&!this->mesh_loader_.LoadedMaterials[0].map_Kd.empty()) {
			rec.data.hasTexture = true;
			rec.data.texture = this->cuda_texture_objects_[meshID];
		}
		else {
			rec.data.hasTexture = false;
		}
		rec.data.index = (gdt::vec3i*)(thrust::raw_pointer_cast(this->index_buffers_[meshID].data()));
		rec.data.vertex = (gdt::vec3f*)(thrust::raw_pointer_cast(this->vertex_buffers_[meshID].data()));
		rec.data.normal = (gdt::vec3f*)thrust::raw_pointer_cast(this->normal_buffers_[meshID].data());
		rec.data.texcoord = (gdt::vec2f*)thrust::raw_pointer_cast(this->texcoord_buffers_[meshID].data());
		hitgroupRecords.push_back(rec);
	}
	this->hitgroup_records_buffer_ = thrust::device_malloc<HitgroupRecord>(hitgroupRecords.size());
	cudaMemcpy(thrust::raw_pointer_cast(this->hitgroup_records_buffer_), hitgroupRecords.data(), hitgroupRecords.size() * sizeof(HitgroupRecord), cudaMemcpyHostToDevice);
	HitgroupRecord* hitgroup_records_buffer_pointer = thrust::raw_pointer_cast(this->hitgroup_records_buffer_);

	this->optix_shader_binding_table_.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_records_buffer_pointer);
	this->optix_shader_binding_table_.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
	this->optix_shader_binding_table_.hitgroupRecordCount = (int)this->optix_hitgroups_.size();
}

void ucgh::RayHandler::createTextures()
{
	int numTextures = textures_.size();

	this->texture_arrays_.resize(numTextures);
	this->cuda_texture_objects_.resize(numTextures);

	for (int textureID = 0; textureID < numTextures; textureID++) {
		auto texture = this->textures_[textureID];

		cudaResourceDesc res_desc = {};

		cudaChannelFormatDesc channel_desc;
		int32_t width = texture.cols;
		int32_t height = texture.rows;
		int32_t numComponents = 4;
		int32_t pitch = width * numComponents * sizeof(uint8_t);
		channel_desc = cudaCreateChannelDesc<uchar4>();

		cudaArray_t& pixelArray = this->texture_arrays_[textureID];
		CUDA_CHECK(MallocArray(&pixelArray,
			&channel_desc,
			width, height));

		CUDA_CHECK(Memcpy2DToArray(pixelArray,
			/* offset */0, 0,
			texture.data,
			pitch, pitch, height,
			cudaMemcpyHostToDevice));

		res_desc.resType = cudaResourceTypeArray;
		res_desc.res.array.array = pixelArray;

		cudaTextureDesc tex_desc = {};
		tex_desc.addressMode[0] = cudaAddressModeWrap;
		tex_desc.addressMode[1] = cudaAddressModeWrap;
		tex_desc.filterMode = cudaFilterModeLinear;
		tex_desc.readMode = cudaReadModeNormalizedFloat;
		tex_desc.normalizedCoords = 1;
		tex_desc.maxAnisotropy = 1;
		tex_desc.maxMipmapLevelClamp = 99;
		tex_desc.minMipmapLevelClamp = 0;
		tex_desc.mipmapFilterMode = cudaFilterModePoint;
		tex_desc.borderColor[0] = 1.0f;
		tex_desc.sRGB = 0;

		// Create texture object
		cudaTextureObject_t cuda_tex = 0;
		CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
		this->cuda_texture_objects_[textureID] = cuda_tex;
	}
}

void ucgh::RayHandler::clearOptixPipeline()
{
	optixPipelineDestroy(this->optix_pipeline_);
}

void ucgh::RayHandler::clearOptixContext()
{
	optixDeviceContextDestroy(this->optix_context_);
}

void ucgh::RayHandler::clearOptixModule()
{
	optixModuleDestroy(this->optix_module_);
}

void ucgh::RayHandler::clearProgramGroups()
{
	for (auto& pg : optix_raygens_) {
		optixProgramGroupDestroy(pg);
	}
	for (auto& pg : optix_misss_) {
		optixProgramGroupDestroy(pg);
	}
	for (auto& pg : optix_hitgroups_) {
		optixProgramGroupDestroy(pg);
	}
}

void ucgh::RayHandler::clearTextures()
{
	for (auto& texture : this->cuda_texture_objects_) {
		cudaDestroyTextureObject(texture);
	}
	for (auto& texture_arraay : this->texture_arrays_) {
		cudaFreeArray(texture_arraay);
	}
}

void ucgh::RayHandler::clearObjects()
{
	this->mesh_data_.clear();
	this->textures_.clear();
}

void ucgh::RayHandler::clearOptixBuffer()
{
	for (auto& vertex : this->vertex_buffers_) {
		vertex.clear();
	}
	for (auto& normal : this->normal_buffers_) {
		normal.clear();
	}
	for (auto& texcoord : this->texcoord_buffers_) {
		texcoord.clear();
	}
	for (auto& index : this->index_buffers_) {
		index.clear();
	}

	this->vertex_buffers_.clear();
	this->normal_buffers_.clear();
	this->texcoord_buffers_.clear();
	this->index_buffers_.clear();
	thrust::device_free(this->raygen_records_buffer_);
	thrust::device_free(this->miss_records_buffer_);
	thrust::device_free(this->hitgroup_records_buffer_);
	thrust::device_free(this->accel_structure_buffer_);
	thrust::device_free(this->launch_params_);
}


OptixTraversableHandle ucgh::RayHandler::buildAccel()
{
	const int numMeshes = this->mesh_data_.size();

	this->vertex_buffers_.resize(numMeshes);
	this->normal_buffers_.resize(numMeshes);
	this->texcoord_buffers_.resize(numMeshes);
	this->index_buffers_.resize(numMeshes);

	OptixTraversableHandle asHandle{ 0 };

	// ==================================================================
	// triangle inputs
	// ==================================================================
	std::vector<OptixBuildInput> triangleInput(numMeshes);
	std::vector<CUdeviceptr> d_vertices(numMeshes);
	std::vector<CUdeviceptr> d_indices(numMeshes);
	std::vector<uint32_t> triangleInputFlags(numMeshes);

	for (int meshID = 0; meshID < numMeshes; meshID++) {
		// upload the model to the device: the builder
		auto& mesh = this->mesh_data_[meshID];
		if (this->vertex_buffers_[meshID].size() != mesh.vertices_.size())
			this->vertex_buffers_[meshID].resize(mesh.vertices_.size());
		cudaMemcpy(thrust::raw_pointer_cast(this->vertex_buffers_[meshID].data()), mesh.vertices_.data(), mesh.vertices_.size() * sizeof(float) * 3, cudaMemcpyHostToDevice);

		this->index_buffers_[meshID].resize(mesh.indices_.size());
		cudaMemcpy(thrust::raw_pointer_cast(this->index_buffers_[meshID].data()), mesh.indices_.data(), mesh.indices_.size() * sizeof(int) * 3, cudaMemcpyHostToDevice);

		if (!mesh.normals_.empty()) {
			if (this->normal_buffers_[meshID].size() != mesh.normals_.size())
				this->normal_buffers_[meshID].resize(mesh.normals_.size());
			cudaMemcpy(thrust::raw_pointer_cast(this->normal_buffers_[meshID].data()), mesh.normals_.data(), mesh.normals_.size() * sizeof(float) * 3, cudaMemcpyHostToDevice);

		}
		if (!mesh.tex_coords_.empty()) {
			if (this->texcoord_buffers_[meshID].size() != mesh.tex_coords_.size())
				this->texcoord_buffers_[meshID].resize(mesh.tex_coords_.size());
			cudaMemcpy(thrust::raw_pointer_cast(this->texcoord_buffers_[meshID].data()), mesh.tex_coords_.data(), mesh.tex_coords_.size() * sizeof(float) * 2, cudaMemcpyHostToDevice);
		}

		triangleInput[meshID] = {};
		triangleInput[meshID].type
			= OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

		// create local variables, because we need a *pointer* to the
		// device pointers
		d_vertices[meshID] = reinterpret_cast<CUdeviceptr>(thrust::raw_pointer_cast(this->vertex_buffers_[meshID].data()));
		d_indices[meshID] = reinterpret_cast<CUdeviceptr>(thrust::raw_pointer_cast(this->index_buffers_[meshID].data()));

		triangleInput[meshID].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangleInput[meshID].triangleArray.vertexStrideInBytes = sizeof(gdt::vec3f);
		triangleInput[meshID].triangleArray.numVertices = (int)mesh.vertices_.size();
		triangleInput[meshID].triangleArray.vertexBuffers = &d_vertices[meshID];

		triangleInput[meshID].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
		triangleInput[meshID].triangleArray.indexStrideInBytes = sizeof(gdt::vec3i);
		triangleInput[meshID].triangleArray.numIndexTriplets = (int)mesh.indices_.size();
		triangleInput[meshID].triangleArray.indexBuffer = d_indices[meshID];

		triangleInputFlags[meshID] = 0;

		// in this example we have one SBT entry, and no per-primitive
		// materials:
		triangleInput[meshID].triangleArray.flags = &triangleInputFlags[meshID];
		triangleInput[meshID].triangleArray.numSbtRecords = 1;
		triangleInput[meshID].triangleArray.sbtIndexOffsetBuffer = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetSizeInBytes = 0;
		triangleInput[meshID].triangleArray.sbtIndexOffsetStrideInBytes = 0;
	}
	// ==================================================================
	// BLAS setup
	// ==================================================================

	OptixAccelBuildOptions accelOptions = {};
	accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE
		| OPTIX_BUILD_FLAG_ALLOW_COMPACTION
		;
	accelOptions.motionOptions.numKeys = 1;
	accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

	OptixAccelBufferSizes blas_buffer_sizes;
	OPTIX_CHECK(optixAccelComputeMemoryUsage
	(this->optix_context_,
		&accelOptions,
		triangleInput.data(),
		(int)numMeshes,  // num_build_inputs
		&blas_buffer_sizes
	));

	// ==================================================================
	// prepare compaction
	// ==================================================================

	thrust::device_ptr<uint64_t> compacted_size_buffer = thrust::device_malloc<uint64_t>(1);

	OptixAccelEmitDesc emit_desc;
	emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emit_desc.result = reinterpret_cast<CUdeviceptr>(thrust::raw_pointer_cast(compacted_size_buffer));

	// ==================================================================
	// execute build (main stage)
	// ==================================================================

	thrust::device_ptr<unsigned char> temp_buffer = thrust::device_malloc(blas_buffer_sizes.tempSizeInBytes);

	thrust::device_ptr<unsigned char> output_buffer = thrust::device_malloc(blas_buffer_sizes.outputSizeInBytes);

	OPTIX_CHECK(optixAccelBuild(this->optix_context_,
		0,
		&accelOptions,
		triangleInput.data(),
		(int)numMeshes,
		reinterpret_cast<CUdeviceptr>(thrust::raw_pointer_cast(temp_buffer)),
		blas_buffer_sizes.tempSizeInBytes,

		reinterpret_cast<CUdeviceptr>(thrust::raw_pointer_cast(output_buffer)),
		blas_buffer_sizes.outputSizeInBytes,

		&asHandle,

		&emit_desc, 1
	));
	CUDA_SYNC_CHECK();
		
	// ==================================================================
	// perform compaction
	// ==================================================================
	uint64_t compacted_size;
	thrust::copy(compacted_size_buffer, compacted_size_buffer + 1, &compacted_size);

	this->accel_structure_buffer_ = thrust::device_malloc<char>(compacted_size);
	OPTIX_CHECK(optixAccelCompact(this->optix_context_,
		/*stream:*/0,
		asHandle,
		reinterpret_cast<CUdeviceptr>(thrust::raw_pointer_cast(accel_structure_buffer_)),
		compacted_size,
		&asHandle));
	CUDA_SYNC_CHECK();

	// ==================================================================
	// aaaaaand .... clean up
	// ==================================================================
	thrust::device_free(temp_buffer);
	thrust::device_free(output_buffer);
	return asHandle;
}

void ucgh::RayHandler::setBufferSize(gdt::vec2i size)
{
	this->framebuffer_ = thrust::device_malloc<float>(size[0] * size[1] * 3);
	this->depthbuffer_ = thrust::device_malloc<float>(size[0] * size[1]);
	this->z_buffer_ = thrust::device_malloc<float>(size[0] * size[1]);
	this->host_framebuffer_.resize(size[0] * size[1] * 3);
	this->launch_params_host_.frame.size = size;
	this->launch_params_host_.frame.color_buffer_r = reinterpret_cast<float*>(thrust::raw_pointer_cast(this->framebuffer_));
	this->launch_params_host_.frame.color_buffer_g = this->launch_params_host_.frame.color_buffer_r + size[0] * size[1];
	this->launch_params_host_.frame.color_buffer_b = this->launch_params_host_.frame.color_buffer_g + size[0] * size[1];
	this->launch_params_host_.frame.depth_buffer = reinterpret_cast<float*>(thrust::raw_pointer_cast(this->depthbuffer_));
	this->launch_params_host_.frame.z_buffer = reinterpret_cast<float*>(thrust::raw_pointer_cast(this->z_buffer_));
}
void ucgh::RayHandler::clearBuffer()
{
	thrust::device_free(this->framebuffer_);
	thrust::device_free(this->depthbuffer_);
	thrust::device_free(this->z_buffer_);
	this->host_framebuffer_.clear();
}
void ucgh::RayHandler::clearAll()
{
	this->clearOptixPipeline();
	this->clearProgramGroups();
	this->clearOptixBuffer();
	this->clearBuffer();
	this->clearObjects();
	this->clearTextures();
	this->clearOptixModule();
	this->clearOptixContext();
}
void ucgh::RayHandler::setCamera(gdt::vec3f position, gdt::vec3f vertical, gdt::vec3f horizontal, gdt::vec3f direction)
{
	this->launch_params_host_.camera.position = position;
	this->launch_params_host_.camera.vertical = vertical;
	this->launch_params_host_.camera.horizontal = horizontal;
	this->launch_params_host_.camera.direction = direction;
}

void ucgh::RayHandler::switchCameraType(ucgh::CameraType camera_type)
{
	this->initOptixRaygen(camera_type);
	this->initOptixPipeline();
	thrust::device_free(this->raygen_records_buffer_);
	thrust::device_free(this->miss_records_buffer_);
	thrust::device_free(this->hitgroup_records_buffer_);
	this->initOptixSBT();
}

