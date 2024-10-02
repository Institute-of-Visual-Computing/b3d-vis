#include <filesystem>

#include <catch2/benchmark/catch_benchmark.hpp>

#include "FitsHelper.h"
#include "catch2/catch_test_macros.hpp"
#include "include/NanoCutApi.h"

const auto testSource = "test_source.fits";
const auto testSourceMask = "test_source_mask.fits";

static auto setUp() -> void
{
	const auto data = std::vector<float>{2.5f};
	const auto maskData = std::vector<long>{ 1 };

	constexpr auto boxSize = Vec3I{1,1,1};
	writeFitsFile(testSource, boxSize, data);
	writeFitsFile(testSourceMask, boxSize, maskData);
}

TEST_CASE("NanoCutApiTests", "[apply fits source and mask and store as nanoVDB]")
{
	setUp();

	const auto source = testSource;
	const auto mask = testSourceMask;

	auto result = ncConvertFitsToNanoVdbWithMask(source, mask, "test.nvdb");

	if (!std::filesystem::exists(source) or !std::filesystem::exists(mask))
	{
		REQUIRE(result == NANOCUT_INVALIDE_ARGUMENT);
	}
	else
	{
		REQUIRE(result == NANOCUT_OK);
	}

	BENCHMARK("n4565 benchmark")
	{
		ncConvertFitsToNanoVdbWithMask(source, mask, "test.nvdb");
	};
}
