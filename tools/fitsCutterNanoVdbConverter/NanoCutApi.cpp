#include "include/NanoCutApi.h"

#include <ranges>
#include <nanovdb/util/IO.h>
#include "FitsHelper.h"

auto ncConvertFitsToNanoVdbWithMask(const char* fitsFile, const char* muskFile, const char* destinationNanoVdbFile) -> ncResult
{
	const auto muskFilePath = std::filesystem::path{ muskFile };
	const auto fitsFilePath = std::filesystem::path{ fitsFile };
	const auto destinationPath = std::filesystem::path{ destinationNanoVdbFile };

	if(std::filesystem::exists(muskFilePath) || std::filesystem::exists(fitsFilePath) || destinationNanoVdbFile == nullptr)
	{
		return NANOCUT_INVALIDE_ARGUMENT;
	}

	const auto map = extractPerClusterBox(muskFilePath, Box3I::maxBox(), Vec3I{});

	auto clusters = std::vector<ClusterId>{};
	clusters.reserve(map.size());

	for (const auto& cluster : map | std::views::keys)
	{
		clusters.push_back(cluster);
	}

	const auto maskData = extractBinaryClusterMask(muskFilePath, clusters, Box3I::maxBox());
	const auto sourceData = extractData(fitsFilePath, Box3I::maxBox());

	//TODO: maybe we should pass it over parameter, or select a "good" value??
	constexpr auto maskedValue = -100.0f;
	const auto filteredData = applyMask(sourceData.data, maskData, maskedValue);

	const auto generatedNanoVdb = generateNanoVdb(sourceData.box.size(), maskedValue, 0.0f, filteredData);

	try
	{
		nanovdb::io::writeGrid(destinationPath.generic_string(), generatedNanoVdb,
		nanovdb::io::Codec::NONE); // TODO: enable nanovdb::io::Codec::BLOSC
	}
	catch (...)
	{
		return NANOCUT_INTERNAL_ERROR;
	}
	

	return NANOCUT_OK;
}
