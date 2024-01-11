#include "NanoCutterParser.h"


#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>


auto cutterParser::load(const std::filesystem::path& file) -> TreeNode
{
	assert(std::filesystem::exists(file));
	std::ifstream f(file);

	TreeNode root;

	try
	{
		const auto data = json::parse(f);
		root = data.get<TreeNode>();
	}
	catch (json::type_error& e)
	{
		std::cout << e.what();
		// [json.exception.type_error.304] cannot use at() with object
	}
	return root;
}

auto cutterParser::store(const std::filesystem::path& file, const TreeNode& root) -> void
{
	// assert(std::filesystem::exists(file));
	const auto j = json( root );
	std::ofstream o(file);
	o << std::setw(4) << j << std::endl;
}
