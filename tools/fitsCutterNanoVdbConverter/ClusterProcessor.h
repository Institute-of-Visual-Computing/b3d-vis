#pragma once

#include <NanoCutterParser.h>
#include <cstdint>
#include <filesystem>


#include "Common.h"

struct Downsampler
{
};

struct ProcessorResult
{
	ClusterId clusterId;
	uint64_t totalClusterMemorySize;
	cutterParser::TreeNode clusterNode;
};

template<typename DownsamplerType = Downsampler>
class ClusterProcessor
{
public:
	ClusterProcessor(const std::filesystem::path& sourceFile, const std::filesystem::path& maskFile,
					 const std::filesystem::path& storageLocation)
		: sourceFile_(sourceFile), maskFile_(maskFile), storageLocation_(storageLocation)
	{
	}

	virtual ~ClusterProcessor()
	{
	}

	[[nodiscard]] virtual auto process(const ClusterId clusterId, const Box3I& rootBox) -> ProcessorResult = 0;

protected:
	std::filesystem::path sourceFile_;
	std::filesystem::path maskFile_;
	std::filesystem::path storageLocation_;

	DownsamplerType downsampler_{};
};
