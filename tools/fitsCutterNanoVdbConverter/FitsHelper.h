#pragma once

#include <filesystem>
#include <map>

#include <mdspan>

#include "Common.h"

using ClusterId = int;

[[nodiscard]] auto extractPerClusterBox(const std::filesystem::path& srcFile, const Box3I& searchBox = Box3I::maxBox(),
										const Vec3I& perBatchSearchSize = Vec3I{}) -> std::map<ClusterId, Box3I>;

[[nodiscard]] auto extractClusterMask(const std::filesystem::path& file, std::vector<ClusterId> clusters,
									  const Box3I& searchBox = Box3I::maxBox()) -> std::vector<uint32_t>;

// all selected clusters will be merged to a single bit
[[nodiscard]] auto extractBinaryClusterMask(const std::filesystem::path& file, std::vector<ClusterId> clusters,
											const Box3I& searchBox = Box3I::maxBox()) -> std::vector<bool>;

template <typename T1, typename T2>
[[nodiscard]] auto extractData(const std::filesystem::path& file, const Box3I& searchBox = Box3I::maxBox(),
							   const std::vector<T2>& mask = {}) -> std::vector<T1>;
//template <typename T>
//[[nodiscard]] auto extractData(const std::filesystem::path& file, const Box3I& searchBox = Box3I::maxBox()) -> std::vector<T>;


[[nodiscard]] auto extractData(const std::filesystem::path& file, const Box3I& searchBox = Box3I::maxBox()) -> std::vector<float>;

auto applyMask(std::vector<float>& data, const std::vector<bool>& mask);
//
//template <typename T>
//inline auto extractData(const std::filesystem::path& file, const Box3I& searchBox) -> std::vector<T>
//{
//	return std::vector<T>();
//}
