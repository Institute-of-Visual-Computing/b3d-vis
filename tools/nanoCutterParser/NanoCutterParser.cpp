#include "NanoCutterParser.h"


#include <cassert>
#include <fstream>
#include <iostream>


auto cutterParser::load(const std::filesystem::path& file) -> B3DDataSet
{
	assert(std::filesystem::exists(file));
	std::ifstream f(file);

	auto dataSet = B3DDataSet{};

	try
	{
		const auto data = json::parse(f);
		dataSet = data.get<B3DDataSet>();
	}
	catch (json::type_error& e)
	{
		std::cout << e.what();
		// [json.exception.type_error.304] cannot use at() with object
	}
	return dataSet;
}

auto cutterParser::store(const std::filesystem::path& file, const B3DDataSet& dataSet) -> void
{
	// assert(std::filesystem::exists(file));
	const auto j = json( dataSet );
	std::ofstream o(file);
	o << std::setw(4) << j << std::endl;
}
