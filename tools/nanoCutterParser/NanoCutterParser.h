#pragma once
#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <string>
#include <vector>


namespace cutterParser
{
	enum class PartitionStrategy
	{
		binary
	};
	NLOHMANN_JSON_SERIALIZE_ENUM(PartitionStrategy, { { PartitionStrategy::binary, "binary" } })

	enum class DownsamplingStrategy
	{
		min,
		max,
		mean
	};

	NLOHMANN_JSON_SERIALIZE_ENUM(DownsamplingStrategy,
								 { { DownsamplingStrategy::min, "min" },
								   { DownsamplingStrategy::max, "max" },
								   { DownsamplingStrategy::mean, "mean" } })

	struct Box
	{
		std::array<float, 3> min;
		std::array<float, 3> max;
		NLOHMANN_DEFINE_TYPE_INTRUSIVE(Box, min, max);
	};

	struct TreeNode
	{
		std::string nanoVdbFile{};
		std::size_t nanoVdbBufferSize{};
		Box aabb{};
		std::vector<TreeNode> children{};
		float minValue;
		float maxValue;
		NLOHMANN_DEFINE_TYPE_INTRUSIVE(TreeNode, nanoVdbFile, nanoVdbBufferSize, aabb, children, minValue, maxValue)
	};

	struct Cluster
	{
		uint64_t clusterId;
		uint64_t accelerationStructureTotalMemorySize;
		TreeNode accelerationStructureRoot;
		NLOHMANN_DEFINE_TYPE_INTRUSIVE(Cluster, clusterId, accelerationStructureTotalMemorySize, accelerationStructureRoot);
	};

	struct B3DDataSet
	{
		std::string sourceFitsFile;
		uint64_t totalMemorySize;
		Box setBox;

		PartitionStrategy partition;
		std::vector<Cluster> clusters;

		NLOHMANN_DEFINE_TYPE_INTRUSIVE(B3DDataSet, sourceFitsFile, totalMemorySize, setBox, partition, clusters);
	};

	using json = nlohmann::json;

	auto load(const std::filesystem::path& file) -> B3DDataSet;
	auto store(const std::filesystem::path& file, const B3DDataSet& dataSet) -> void;
} // namespace cutterParser
