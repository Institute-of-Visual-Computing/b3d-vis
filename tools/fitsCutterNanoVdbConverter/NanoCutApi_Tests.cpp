#include <filesystem>

#include "catch2/catch_test_macros.hpp"
#include "include/NanoCutApi.h"


TEST_CASE("NanoCutApiTests", "[apply fits source and mask and store as nanoVDB]")
{

	const auto source = "D:/datacubes/n4565/n4565_lincube_big_downsampled.raw";
	const auto mask = "D:/datacubes/n4565/n4565_lincube_big_mask_downsampled.raw";

	auto result = ncConvertFitsToNanoVdbWithMask(source, mask, "test.nvdb");

	if (!std::filesystem::exists(source) or !std::filesystem::exists(mask))
	{
		REQUIRE(result == NANOCUT_INVALIDE_ARGUMENT);
	}
	else
	{
		REQUIRE(result == NANOCUT_OK);
	}
}
