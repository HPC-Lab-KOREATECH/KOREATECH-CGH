#pragma once

#include <filesystem>
#include "../argparse/argparse.hpp"
#include "json.hpp"
#include <iostream>
using json = nlohmann::json;

namespace ucgh {
	namespace parser {
		class Argument {
		public:
			std::string argument_name_;
			std::string help_message_;
			char scan_type_;
			int num_of_data_;
			bool is_required_;
			bool is_implicit_;
			void set(std::string argument_name, std::string help_message, char scan_type = 0, int num_of_data = 0, bool is_required=false, bool is_implicit=false);
			Argument() {}
			Argument(std::string argument_name, std::string help_message, char scan_type = 0, int num_of_data = 0, bool is_required = false, bool is_implicit = false);
			argparse::Argument& addArgument(argparse::ArgumentParser* command);
		};
		class ParserExecuter {
		public:
			argparse::ArgumentParser* command_;
			std::string command_name_;
			std::vector<ucgh::parser::Argument> arguments_list_;
			template <typename T> T getJsonValue(std::string argument_name, json& json_data);
			template <typename T> T getArguValue(std::string argument_name);
			template <typename T> T getValue(std::string argument_name, json& json_data);
			virtual void addArgumentList(std::string argument_name, std::string help_message, char scan_type = 0, int num_of_data = 0, bool is_required = false, bool is_implicit = false);
			virtual void initArguments();
			virtual bool isUsedInArgument(std::string argument_name);

			virtual void set() = 0;
		};
	};
};