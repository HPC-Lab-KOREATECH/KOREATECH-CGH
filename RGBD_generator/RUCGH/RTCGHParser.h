#pragma once
#include "ParsingManager.h"

namespace ucgh {

	class RUCGHParams {
	public:
		std::string precision;
		size_t width;
		size_t height;
		double pixel_pitch;
		std::vector<double> wavelength;

		double object_mindepth;
		double object_maxdepth;

		std::string rgb_output_path;
		std::string depth_output_path;
		std::string csv_path;
		std::string obj_path;
		std::string ptx_path;
		std::string floor_texture_path;

		bool is_loadmode;
		bool is_rgbdonly;

		size_t begin_index;
		size_t end_index;

		size_t num_objects;
		size_t device_number;
		size_t num_devices;
		void print();
	};

	namespace parser {
		class GenerationParser : public ucgh::parser::ParserExecuter
		{
		public:
			GenerationParser();
			virtual void set();
			virtual RUCGHParams makeParams();
		};
	};
}
