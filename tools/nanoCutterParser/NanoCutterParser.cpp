#include "NanoCutterParser.h"


#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>


auto cutterParser::load(const std::filesystem::path& file) -> std::vector<TreeNode>
{
	assert(std::filesystem::exists(file));
	std::ifstream f(file);

	auto trees = std::vector<TreeNode>{};

	try
	{
		const auto data = json::parse(f);
		trees = data.get<std::vector<TreeNode>>();
	}
	catch (json::type_error& e)
	{
		std::cout << e.what();
		// [json.exception.type_error.304] cannot use at() with object
	}
	return trees;
}

auto cutterParser::store(const std::filesystem::path& file, const std::vector<TreeNode>& trees) -> void
{
	// assert(std::filesystem::exists(file));
	const auto j = json( trees );
	std::ofstream o(file);
	o << std::setw(4) << j << std::endl;
}
