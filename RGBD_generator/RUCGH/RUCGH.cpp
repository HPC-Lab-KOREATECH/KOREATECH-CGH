#include "RUCGH.h"

struct CsvRow {
    std::string label;
    std::array<double, 9> values;
};

namespace {
bool isTextureExtension(const std::filesystem::path& path)
{
    auto ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg";
}

std::filesystem::path findObjectTexture(const std::filesystem::path& object_dir, const std::filesystem::path& obj_path)
{
    auto preferred_texture = object_dir / "materials" / "textures" / "texture.png";
    if (std::filesystem::exists(preferred_texture)) {
        return preferred_texture;
    }

    auto texture_dir = object_dir / "materials" / "textures";
    if (std::filesystem::exists(texture_dir) && std::filesystem::is_directory(texture_dir)) {
        for (const auto& texture_entry : std::filesystem::directory_iterator(texture_dir)) {
            if (std::filesystem::is_regular_file(texture_entry) && isTextureExtension(texture_entry.path())) {
                return texture_entry.path();
            }
        }
    }

    for (const auto& texture_entry : std::filesystem::directory_iterator(obj_path.parent_path())) {
        if (std::filesystem::is_regular_file(texture_entry) && isTextureExtension(texture_entry.path())) {
            return texture_entry.path();
        }
    }

    return {};
}
}

std::vector<CsvRow> load_csv(const std::string& filename) {
    std::vector<CsvRow> rows;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }

    std::string line;
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;

        CsvRow row;
        if (!std::getline(ss, token, ',')) {
            throw std::runtime_error("Missing label in row");
        }
        row.label = token;

        for (size_t i = 0; i < 9; ++i) {
            if (!std::getline(ss, token, ',')) {
                throw std::runtime_error("Missing value in row: " + line);
            }
            row.values[i] = std::stod(token);
        }

        rows.push_back(row);
    }

    return rows;
}

template <ucgh::floating_point T>
void ucgh::RUCGH<T>::setRayHandler(gdt::vec2i size, int device_id, std::filesystem::path ptx_path)
{
    this->ray_handler_->init(device_id, ptx_path);
    this->ray_handler_->setBufferSize(size);
}

template<ucgh::floating_point T>
void ucgh::RUCGH<T>::loadMultipleObjects(std::filesystem::path const& file_directory, std::vector<double> const& boundary, size_t num_items, std::string& record)
{
    auto scale_factor = std::max(this->param_.nx_, this->param_.ny_);
    std::array<double, 2> xy_size = { -(double)this->param_.nx_ / 2.0, (double)this->param_.nx_ / 2.0 };

    std::random_device device;
    std::default_random_engine el(device());
    std::uniform_real_distribution<double> uniform_dist_scale(scale_factor * 0.2, scale_factor * 0.3);
    std::uniform_real_distribution<double> uniform_dist_xy(xy_size[0], xy_size[1]);
    std::uniform_real_distribution<double> uniform_dist_angle(-2.0 * std::numbers::pi, 2.0 * std::numbers::pi);

    std::vector<double> depth_dist(num_items);
    std::iota(depth_dist.begin(), depth_dist.end(), 1);
    std::for_each(depth_dist.begin(), depth_dist.end(), [boundary, num_items](auto& ele) { ele = ele / num_items * (boundary[1] - boundary[0]) + boundary[0]; });
    std::shuffle(depth_dist.begin(), depth_dist.end(), el);

    int sqrt_num_items = std::sqrt(num_items);
    std::vector<double> x_dist(num_items);
    std::vector<double> y_dist(num_items);
    if (num_items == 1) {
        x_dist[0] = 0.0;
        y_dist[0] = 0.0;
    }
    else if (sqrt_num_items > 1) {
        std::iota(x_dist.begin(), x_dist.end(), 0);
        std::iota(y_dist.begin(), y_dist.end(), 0);
        std::for_each(x_dist.begin(), x_dist.begin() + sqrt_num_items * sqrt_num_items, [sqrt_num_items, xy_size](auto& ele) {
            ele = ((int)ele % sqrt_num_items) / (double)(sqrt_num_items - 1) * (xy_size[1] - xy_size[0]) + xy_size[0];
            });
        std::for_each(y_dist.begin(), y_dist.begin() + sqrt_num_items * sqrt_num_items, [sqrt_num_items, xy_size](auto& ele) {
            ele = ((int)ele / sqrt_num_items) / (double)(sqrt_num_items - 1) * (xy_size[1] - xy_size[0]) + xy_size[0];
            });
        std::for_each(x_dist.begin() + sqrt_num_items * sqrt_num_items, x_dist.end(), [&uniform_dist_xy, &el](auto& ele) {
            ele = uniform_dist_xy(el);
            });
        std::for_each(y_dist.begin() + sqrt_num_items * sqrt_num_items, y_dist.end(), [&uniform_dist_xy, &el](auto& ele) {
            ele = uniform_dist_xy(el);
            });
    }
    else {
        std::generate(x_dist.begin(), x_dist.end(), [&uniform_dist_xy, &el]() { return uniform_dist_xy(el); });
        std::generate(y_dist.begin(), y_dist.end(), [&uniform_dist_xy, &el]() { return uniform_dist_xy(el); });
    }
    std::shuffle(x_dist.begin(), x_dist.end(), el);
    std::shuffle(y_dist.begin(), y_dist.end(), el);

    auto x_dist_iter = x_dist.begin();
    auto y_dist_iter = y_dist.begin();
    auto depth_dist_iter = depth_dist.begin();

    std::vector<std::filesystem::path> directories;
    for (const auto& dir_entry : std::filesystem::directory_iterator(file_directory))
    {
        if (std::filesystem::is_directory(dir_entry))
            directories.push_back(dir_entry.path());
    }

    std::shuffle(directories.begin(), directories.end(), el);

    this->ray_handler_->mesh_data_.reserve(num_items);
    this->ray_handler_->textures_.reserve(num_items);
    record = "name, scale_x, scale_y, scale_z, yaw, pitch, roll, translation_x, translation_y, translation_z\n";

    for (size_t i = 0; i < num_items; i++)
    {
        auto dir_path = directories[i];
        std::filesystem::path meshes_dir = dir_path / "meshes";
        if (!std::filesystem::exists(meshes_dir) || !std::filesystem::is_directory(meshes_dir)) {
            num_items += 1;
            continue;
        }
        bool has_obj_file = false;
        for (const auto& mesh_entry : std::filesystem::directory_iterator(meshes_dir))
        {
            if (mesh_entry.path().extension() == ".obj")
            {
                has_obj_file = true;
                std::filesystem::path obj_path = mesh_entry.path();
                std::filesystem::path mtl_path = obj_path.parent_path() / (obj_path.stem().string() + ".mtl");
                std::filesystem::path texture_path = findObjectTexture(dir_path, obj_path);

                std::cout << "Loading: " << obj_path << ", Material: " << mtl_path << ", Texture: " << texture_path << std::endl;

                auto scale_ratio = uniform_dist_scale(el);
                std::array<double, 3> scale = { scale_ratio, scale_ratio, scale_ratio };
                std::array<double, 3> rotation = { uniform_dist_angle(el), uniform_dist_angle(el), uniform_dist_angle(el) };
                std::array<double, 3> translation = { *x_dist_iter++, *y_dist_iter++, *depth_dist_iter++ / this->param_.dx_ };
                std::array<double, 3> translation_out = { translation[0] / this->param_.nx_, translation[1] / this->param_.nx_, translation[2] / this->param_.nx_ };

                if (!std::isfinite(translation[0]) || !std::isfinite(translation[1]) || !std::isfinite(translation[2])) {
                    throw std::runtime_error("Non-finite object translation generated for " + dir_path.string());
                }

                this->ray_handler_->loadObj(obj_path, mtl_path, texture_path, scale, rotation, translation);

                for (double& value : translation_out) {
                    if ((value < std::numeric_limits<float>::min() && value > -std::numeric_limits<float>::min()) ||
                        value < -std::numeric_limits<float>::max() || value > std::numeric_limits<float>::max()) {
                        value = 0.0;
                    }
                }

                record = record + std::format("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n", dir_path.stem().string(), scale[0] / this->param_.nx_, scale[1] / this->param_.nx_, scale[2] / this->param_.nx_,
                    rotation[0], rotation[1], rotation[2], translation_out[0], translation_out[1], translation_out[2]);
                break;
            }
        }
        if (!has_obj_file) {
            num_items += 1;
        }
    }
}

template<ucgh::floating_point T>
void ucgh::RUCGH<T>::loadMultipleObjects(std::filesystem::path const& file_directory, std::filesystem::path const& record_path)
{
    auto csvs = load_csv(record_path.string());

    for (auto& row : csvs)
    {
        auto meshes_dir = file_directory / row.label / "meshes";
        if (!std::filesystem::exists(meshes_dir) || !std::filesystem::is_directory(meshes_dir))
            continue;

        for (const auto& mesh_entry : std::filesystem::directory_iterator(meshes_dir))
        {
            if (mesh_entry.path().extension() == ".obj")
            {
                std::filesystem::path obj_path = mesh_entry.path();
                std::filesystem::path mtl_path = obj_path.parent_path() / (obj_path.stem().string() + ".mtl");
                std::filesystem::path texture_path = findObjectTexture(file_directory / row.label, obj_path);

                std::array<double, 3> scale = { row.values[0] * this->param_.nx_, row.values[1] * this->param_.nx_, row.values[2] * this->param_.nx_ };
                std::array<double, 3> rotation = { row.values[3], row.values[4], row.values[5] };
                std::array<double, 3> translation = { row.values[6] * this->param_.nx_, row.values[7] * this->param_.nx_, row.values[8] * this->param_.nx_ };

                this->ray_handler_->loadObj(obj_path, mtl_path, texture_path, scale, rotation, translation);
            }
        }
    }
}

template<ucgh::floating_point T>
void ucgh::RUCGH<T>::createPlane(float max_z, std::filesystem::path plane_texture_dir, std::string& record)
{
    ucgh::Mesh mesh;
    mesh.resize(4);

    float coord_x = this->param_.nx_;
    float coord_y = this->param_.ny_;

    mesh.vertices_[0] = gdt::vec3f(-coord_x, -coord_y, max_z / this->param_.dx_);
    mesh.vertices_[1] = gdt::vec3f(coord_x, -coord_y, max_z / this->param_.dx_);
    mesh.vertices_[2] = gdt::vec3f(coord_x, coord_y, max_z / this->param_.dx_);
    mesh.vertices_[3] = gdt::vec3f(-coord_x, coord_y, max_z / this->param_.dx_);

    for (int i = 0; i < 4; ++i) {
        mesh.normals_[i] = gdt::vec3f(0.0f, 0.f, -1.0f);
    }

    mesh.tex_coords_[0] = gdt::vec2f(0.0f, 0.0f);
    mesh.tex_coords_[1] = gdt::vec2f(1.0f, 0.0f);
    mesh.tex_coords_[2] = gdt::vec2f(1.0f, 1.0f);
    mesh.tex_coords_[3] = gdt::vec2f(0.0f, 1.0f);

    mesh.indices_.push_back(gdt::vec3i(0, 1, 2));
    mesh.indices_.push_back(gdt::vec3i(0, 2, 3));
    mesh.mesh_name_.push_back("plane");

    this->ray_handler_->mesh_data_.push_back(mesh);

    std::random_device device;
    std::default_random_engine el(device());

    std::vector<std::filesystem::path> texture_files;
    for (const auto& dir_entry : std::filesystem::directory_iterator(plane_texture_dir))
    {
        if (std::filesystem::is_regular_file(dir_entry))
            texture_files.push_back(dir_entry.path());
    }

    std::shuffle(texture_files.begin(), texture_files.end(), el);
    auto selected_texture = texture_files[0];
    auto texture = cv::imread(selected_texture.string());
    cv::cvtColor(texture, texture, cv::COLOR_BGR2RGBA);
    this->ray_handler_->textures_.push_back(texture);

    record = record + std::format("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n", selected_texture.stem().string(), 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

template<ucgh::floating_point T>
void ucgh::RUCGH<T>::loadPlane(float max_z, std::filesystem::path plane_texture_dir, std::filesystem::path record)
{
    ucgh::Mesh mesh;
    mesh.resize(4);

    float coord_x = this->param_.nx_;
    float coord_y = this->param_.ny_;

    mesh.vertices_[0] = gdt::vec3f(-coord_x, -coord_y, max_z / this->param_.dx_);
    mesh.vertices_[1] = gdt::vec3f(coord_x, -coord_y, max_z / this->param_.dx_);
    mesh.vertices_[2] = gdt::vec3f(coord_x, coord_y, max_z / this->param_.dx_);
    mesh.vertices_[3] = gdt::vec3f(-coord_x, coord_y, max_z / this->param_.dx_);

    for (int i = 0; i < 4; ++i) {
        mesh.normals_[i] = gdt::vec3f(0.0f, 0.f, -1.0f);
    }

    mesh.tex_coords_[0] = gdt::vec2f(0.0f, 0.0f);
    mesh.tex_coords_[1] = gdt::vec2f(1.0f, 0.0f);
    mesh.tex_coords_[2] = gdt::vec2f(1.0f, 1.0f);
    mesh.tex_coords_[3] = gdt::vec2f(0.0f, 1.0f);

    mesh.indices_.push_back(gdt::vec3i(0, 1, 2));
    mesh.indices_.push_back(gdt::vec3i(0, 2, 3));
    mesh.mesh_name_.push_back("plane");

    this->ray_handler_->mesh_data_.push_back(mesh);

    auto csvs = load_csv(record.string());
    auto texture_filename = csvs.back().label;
    auto texture = cv::imread((plane_texture_dir / texture_filename).string() + ".png");
    cv::cvtColor(texture, texture, cv::COLOR_BGR2RGBA);
    this->ray_handler_->textures_.push_back(texture);
}

template class ucgh::RUCGH<float>;
