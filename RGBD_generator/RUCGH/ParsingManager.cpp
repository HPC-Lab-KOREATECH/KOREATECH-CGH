#include "ParsingManager.h"


template<typename T>
T ucgh::parser::ParserExecuter::getJsonValue(std::string argument_name, json& json_data)
{
	return json_data[argument_name];
}

template<typename T>
T ucgh::parser::ParserExecuter::getArguValue(std::string argument_name)
{
	std::string dash_argument_name = "--" + argument_name;
	if (this->command_->is_used(dash_argument_name)) {
		return this->command_->get<T>(dash_argument_name);
	}
	else {
		return T();
	}
}

template std::vector<std::string> ucgh::parser::ParserExecuter::getArguValue(std::string argument_name);
template bool ucgh::parser::ParserExecuter::getArguValue(std::string argument_name);
template std::string ucgh::parser::ParserExecuter::getArguValue(std::string argument_name);
template double ucgh::parser::ParserExecuter::getArguValue(std::string argument_name);
template int ucgh::parser::ParserExecuter::getArguValue(std::string argument_name);
template std::vector<double> ucgh::parser::ParserExecuter::getArguValue(std::string argument_name);
template std::vector<int> ucgh::parser::ParserExecuter::getArguValue(std::string argument_name);


template<typename T>
T ucgh::parser::ParserExecuter::getValue(std::string argument_name, json &json_data)
{
	std::string dash_argument_name = "--" + argument_name;
	if (this->command_->is_used(dash_argument_name)) {
		return this->command_->get<T>(dash_argument_name);
	}
	else {
		return this->getJsonValue<T>(argument_name, json_data);
	}
}

template bool ucgh::parser::ParserExecuter::getValue(std::string argument_name, json& json_data);
template int ucgh::parser::ParserExecuter::getValue(std::string argument_name, json& json_data);
template double ucgh::parser::ParserExecuter::getValue(std::string argument_name, json& json_data);
template std::string ucgh::parser::ParserExecuter::getValue(std::string argument_name, json& json_data);
template std::vector<double> ucgh::parser::ParserExecuter::getValue(std::string argument_name, json& json_data);
template std::vector<int> ucgh::parser::ParserExecuter::getValue(std::string argument_name, json& json_data);
template std::vector<std::string> ucgh::parser::ParserExecuter::getValue(std::string argument_name, json& json_data);



void ucgh::parser::ParserExecuter::addArgumentList(std::string argument_name, std::string help_message, char scan_type, int num_of_data, bool is_required, bool is_implicit)
{
	ucgh::parser::Argument argument(argument_name, help_message, scan_type, num_of_data, is_required, is_implicit);
	this->arguments_list_.push_back(argument);
}

void ucgh::parser::ParserExecuter::initArguments()
{
	for (size_t i = 0; i < arguments_list_.size(); i++)
	{
		arguments_list_[i].addArgument(this->command_);
	}
}

bool ucgh::parser::ParserExecuter::isUsedInArgument(std::string argument_name)
{
	std::string dash_argument_name = "--" + argument_name;
	return this->command_->is_used(dash_argument_name);
}

void ucgh::parser::Argument::set(std::string argument_name, std::string help_message, char scan_type, int num_of_data, bool is_required, bool is_implicit)
{
	argument_name_ = argument_name;
	help_message_ = help_message;
	this->scan_type_ = scan_type;
	this->num_of_data_ = num_of_data;
	this->is_required_ = is_required;
	this->is_implicit_ = is_implicit;
}

ucgh::parser::Argument::Argument(std::string argument_name, std::string help_message, char scan_type, int num_of_data, bool is_required, bool is_implicit)
{
	this->set(argument_name, help_message, scan_type, num_of_data, is_required, is_implicit);
}

argparse::Argument& ucgh::parser::Argument::addArgument(argparse::ArgumentParser* command)
{
	auto& parse = command->add_argument("--"+argument_name_);
	
	parse.help(help_message_);
	if (is_required_) {
		parse.required();
	}
	if (is_implicit_) {
		parse.implicit_value(true);
		parse.default_value(false);
	}
	if (num_of_data_ > 1) {
		parse.nargs(num_of_data_);
	}
	if (scan_type_ == 'g') {
		parse.scan<'g', double>();
	}
	else if (scan_type_ == 'i') {
		parse.scan<'i', int>();
	}
	return parse;
}
