#pragma once
#include <string>
#include <vector>

#include "owl/common/math/vec.h"

struct SourceRegion;

struct SourceVolumeLoader
{
	static auto extractSourceRegionsFromCatalogueXML(const std::string& filePath,
													 std::vector<SourceRegion>& sourceBoxes) -> size_t;
	static auto loadDataForSources(const std::string& filePath, std::vector<SourceRegion>& sourceBoxes,
								   std::vector<float>& dataBuffer) -> owl::vec3i;
};
