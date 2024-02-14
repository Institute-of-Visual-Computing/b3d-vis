#pragma once

#include <filesystem>
#include <map>

#include "Common.h"
#include "util/CreateNanoGrid.h"


struct MinMaxBounds
{
	float min;
	float max;
};

[[nodiscard]] auto extractBounds(const std::filesystem::path& srcFile) -> Box3I;

[[nodiscard]] auto extractPerClusterBox(const std::filesystem::path& srcFile, const Box3I& searchBox = Box3I::maxBox(),
										const Vec3I& perBatchSearchSize = Vec3I{}) -> std::map<ClusterId, Box3I>;

[[nodiscard]] auto extractClusterMask(const std::filesystem::path& file, std::vector<ClusterId> clusters,
									  const Box3I& searchBox = Box3I::maxBox()) -> std::vector<uint32_t>;

// all selected clusters will be merged to a single bit
[[nodiscard]] auto extractBinaryClusterMask(const std::filesystem::path& file, std::vector<ClusterId> clusters,
											const Box3I& searchBox = Box3I::maxBox()) -> std::vector<bool>;

struct ExtractedData
{
	Box3I box;
	std::vector<float> data;
};

[[nodiscard]] auto extractData(const std::filesystem::path& file, const Box3I& searchBox = Box3I::maxBox()) -> ExtractedData;

[[nodiscard]] auto applyMask(const std::vector<float>& data, const std::vector<bool>& mask, const float maskedValue = 0.0f) -> std::vector<float>;

[[nodiscard]] auto searchMinMaxBounds(const std::vector<float>& data) -> MinMaxBounds;


auto writeFitsFile(const std::filesystem::path& file, const Vec3I boxSize, const std::vector<long>& data)-> void;
auto writeFitsFile(const std::filesystem::path& file, const Vec3I boxSize, const std::vector<float>& data)-> void;


inline auto flattenIndex(const Vec3I boxSize, const uint64_t i, const uint64_t j, const uint64_t k) -> uint64_t
{
	return static_cast<uint64_t>(boxSize.x) * static_cast<uint64_t>(boxSize.y) * i+
			static_cast<uint64_t>(boxSize.x) * j + k;
}

auto generateNanoVdb(const Vec3I boxSize, const float maskedValues, const float emptySpaceValue, const std::vector<float>& data) -> nanovdb::GridHandle<>;
auto generateNanoVdb(const Vec3I boxSize, const float emptySpaceValue, const std::vector<float>& data,
								   const std::function<float(const uint64_t i, const uint64_t j, const uint64_t k)>& f) -> nanovdb::GridHandle<>;

auto upscaleFitsData(const std::filesystem::path& srcFile, const std::filesystem::path& dstFile, const Vec3I& axisScaleFactor, const std::function<float(const float, const Vec3I&, const Vec3I&)>& filter) -> void;
