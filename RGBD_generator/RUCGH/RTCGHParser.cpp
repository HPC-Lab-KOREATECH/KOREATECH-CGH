#include "RTCGHParser.h"

ucgh::parser::GenerationParser::GenerationParser()
{
	this->command_name_ = std::string("gen");
	this->set();
	this->initArguments();
}

void ucgh::parser::GenerationParser::set()
{
	this->command_ = new argparse::ArgumentParser(this->command_name_);

	this->addArgumentList("precision", "--precision [fp32]");
	this->addArgumentList("width", "--width <width>", 'i');
	this->addArgumentList("height", "--height <height>", 'i');
	this->addArgumentList("wavelength", "wavelength <blue> <green> <red>", 'g', 3);
	this->addArgumentList("pixel_pitch", "--pixel_pitch <pixel_pitch of x-coordinates>", 'g', 1);

	this->addArgumentList("object_mindepth", "minimum depth of object scene", 'g');
	this->addArgumentList("object_maxdepth", "maximum depth of object scene", 'g');

	this->addArgumentList("rgb_output_path", "RGB image directory");
	this->addArgumentList("depth_output_path", "Depth image directory");
	this->addArgumentList("obj_path", "mesh data directory");
	this->addArgumentList("csv_path", "scene CSV directory");
	this->addArgumentList("floor_path", "floor texture path");

	this->addArgumentList("load_csv", "loading mode", 0, 0, false, true);
	this->addArgumentList("rgbd_only", "export RGB + depth only", 0, 0, false, true);

	this->addArgumentList("begin_index", "beginning index of data", 'i');
	this->addArgumentList("end_index", "ending index of data", 'i');
	this->addArgumentList("num_objects", "number_of_objects", 'i', 1);

	this->addArgumentList("ptx_path", "ptx_path");
	this->addArgumentList("device", "device number", 'i', 1);
	this->addArgumentList("num_devices", "device number", 'i', 1);
}

ucgh::RUCGHParams ucgh::parser::GenerationParser::makeParams()
{
	RUCGHParams params;
	params.precision = getArguValue<std::string>("precision");
	params.width = getArguValue<int>("width");
	params.height = getArguValue<int>("height");

	params.object_mindepth = getArguValue<double>("object_mindepth");
	params.object_maxdepth = getArguValue<double>("object_maxdepth");

	params.wavelength = getArguValue<std::vector<double>>("wavelength");
	params.pixel_pitch = getArguValue<double>("pixel_pitch");

	params.rgb_output_path = getArguValue<std::string>("rgb_output_path");
	params.depth_output_path = getArguValue<std::string>("depth_output_path");
	params.obj_path = getArguValue<std::string>("obj_path");
	params.csv_path = getArguValue<std::string>("csv_path");
	params.ptx_path = getArguValue<std::string>("ptx_path");
	params.floor_texture_path = getArguValue<std::string>("floor_path");

	params.is_loadmode = getArguValue<bool>("load_csv");
	params.is_rgbdonly = getArguValue<bool>("rgbd_only");

	params.begin_index = getArguValue<int>("begin_index");
	params.end_index = getArguValue<int>("end_index");

	params.num_objects = getArguValue<int>("num_objects");
	params.device_number = getArguValue<int>("device");
	params.num_devices = getArguValue<int>("num_devices");

	if (params.num_devices < 1) {
		params.num_devices = 1;
	}

	return params;
}

void ucgh::RUCGHParams::print()
{
	std::cout << "Precision: " << precision << "\n";
	std::cout << "Width: " << width << ", Height: " << height << "\n";
	std::cout << "Object min depth: " << object_mindepth << ", Object max depth: " << object_maxdepth << "\n";
	std::cout << "Wavelengths: ";
	for (const auto& w : wavelength) std::cout << w << " ";
	std::cout << "\n";

	std::cout << "Pixel Pitch: " << pixel_pitch << "\n";
	std::cout << "RGB Output Path: " << rgb_output_path << "\n";
	std::cout << "Depth Output Path: " << depth_output_path << "\n";
	std::cout << "Object Path: " << obj_path << "\n";
	std::cout << "CSV Path: " << csv_path << "\n";
	std::cout << "Begin Index: " << begin_index << ", End Index: " << end_index << "\n";
	std::cout << "Number of Objects: " << num_objects << "\n";
	std::cout << "CSV loading : " << this->is_loadmode << std::endl;
	std::cout << "RGBD only gen : " << this->is_rgbdonly << std::endl;
}
