#pragma once

#include <filesystem>

#include "ClusterProcessor.h"
#include "Common.h"
#include "FitsHelper.h"
#include "util/IO.h"

template <typename DownsamplerType, int MaxVoxelsPerSplitThreshold>
class BinaryPartitionClusterProcessor final : public ClusterProcessor<DownsamplerType>
{
public:
	BinaryPartitionClusterProcessor(const std::filesystem::path& sourceFile, const std::filesystem::path& maskFile,
									const std::filesystem::path& storageLocation)
		: ClusterProcessor<DownsamplerType>(sourceFile, maskFile, storageLocation)
	{
	}
	[[nodiscard]] auto process(const ClusterId clusterId, const Box3I& rootBox) -> ProcessorResult override;

private:
	uint64_t totalClusterMemorySize_{ 0 };

	struct ProcessInternal
	{
		cutterParser::TreeNode node;
		std::vector<float> data;
	};
	auto processInternal(const ClusterId clusterId, const Box3I& parentBox, int level = 0) -> ProcessInternal
	{
		const auto [width, height, depth] = parentBox.size();

		const auto needSplit = width * height * depth > MaxVoxelsPerSplitThreshold;

		constexpr auto maskedValue = -100.0f;

		ProcessInternal result;
		auto data = std::vector<float>{};
		auto size = Vec3I{};
		if (needSplit)
		{
			//TODO: also other strategies possible, for now we select longest axis
			const auto axis = std::array{ width, height, depth };
			const auto longestAxis = std::ranges::max_element(axis);
			const auto longestAxisIndex = std::distance(axis.begin(), longestAxis);

			const auto splitAxisLength1 =  *longestAxis / 2;
			const auto splitAxisLength2 = *longestAxis - *longestAxis / 2;

			auto splitBox1 = parentBox;
			auto splitBox2 = parentBox;
			if (longestAxisIndex == 0)
			{
				splitBox1.upper.x -= splitAxisLength2;
				splitBox2.lower.x += splitAxisLength1;
			}
			else if (longestAxisIndex == 1)
			{

				splitBox1.upper.y -= splitAxisLength2;
				splitBox2.lower.y += splitAxisLength1;
			}
			else
			{

				splitBox1.upper.z -= splitAxisLength2;
				splitBox2.lower.z += splitAxisLength1;
			}


			const auto mapSplit1 = extractPerClusterBox(this->maskFile_, splitBox1, Vec3I{});
			const auto mapSplit2 = extractPerClusterBox(this->maskFile_, splitBox2, Vec3I{});

			assert(mapSplit1.contains(clusterId) && mapSplit2.contains(clusterId));

			const auto newSplit1Box = mapSplit1.at(clusterId);
			const auto newSplit2Box = mapSplit2.at(clusterId);
			const auto resultSplit1 = processInternal(clusterId, newSplit1Box, level + 1);
			const auto resultSplit2 = processInternal(clusterId, newSplit2Box, level + 1);

			result.node.children.push_back(resultSplit1.node);
			result.node.children.push_back(resultSplit2.node);

			const auto parentBoxSize = parentBox.size();
			const auto downscaledBoxSize = (parentBoxSize + Vec3I{ 1, 1, 1 }) * 0.5;

			auto downsampledData = std::vector<float>{};
			downsampledData.resize(downscaledBoxSize.x * downscaledBoxSize.y * downscaledBoxSize.z);

			for (auto i = 0ull; i < downscaledBoxSize.z; i++)
			{
				for (auto j = 0ull; j < downscaledBoxSize.y; j++)
				{
					for (auto k = 0ull; k < downscaledBoxSize.x; k++)
					{
						const auto s0 = downscaledBoxSize * 2;
						const auto s1 = s0 + Vec3I{ 0, 0, 1 };
						const auto s2 = s0 + Vec3I{ 0, 1, 0 };
						const auto s3 = s0 + Vec3I{ 0, 1, 1 };
						const auto s4 = s0 + Vec3I{ 1, 0, 0 };
						const auto s5 = s0 + Vec3I{ 1, 0, 1 };
						const auto s6 = s0 + Vec3I{ 1, 1, 0 };
						const auto s7 = s0 + Vec3I{ 1, 1, 1 };


						// TODO: replace with proper downsampling strategy, for now we are doing max value
						const auto samplers = std::array{ s0, s1, s2, s3, s4, s5, s6, s7 };

						auto maxValue = std::numeric_limits<float>::min();
						for (const auto& sampler : samplers)
						{
							auto value{ 0.0f };

							if (newSplit1Box.contains(sampler))
							{
								const auto index3D = sampler - newSplit1Box.lower;
								const auto index = flattenIndex(newSplit1Box.size(), index3D.x, index3D.y, index3D.z);
								value = resultSplit1.data[index];
							}
							else if (newSplit2Box.contains(sampler))
							{
								const auto index3D = sampler - newSplit2Box.lower;
								const auto index = flattenIndex(newSplit2Box.size(), index3D.x, index3D.y, index3D.z);
								value = resultSplit2.data[index];
							}

							maxValue = std::max({ value, maxValue });
						}

						const auto dstIndex = flattenIndex(downscaledBoxSize, i, j, k);
						downsampledData[dstIndex] = maxValue;
					}
				}
			}
			data = downsampledData;
			size = downscaledBoxSize;
		}
		else
		{
			const auto mask = extractBinaryClusterMask(this->maskFile_, { clusterId }, parentBox);
			auto extractedData = extractData(this->sourceFile_, parentBox);


			const auto filteredData = applyMask(extractedData.data, mask, maskedValue);
			data = filteredData;
			size = parentBox.size();
		}

#if 0
			const auto fitsFileName = std::format("filtered_cluster_{}_extent_{}_{}_{}.fits", clusterId, size.x, size.y, size.z);
			const auto fitsPath = (this->storageLocation_ / fitsFileName).string();

			writeFitsFile(fitsPath.c_str(), size, filteredData);
#endif

		//TODO: This could be extracted and moved to base class, because it could be useful in other partition methods as well.
		const auto fileName =
			std::format("nano__cluster_{}__level_{}__box_{}_{}_{}-{}_{}_{}.nvdb", clusterId, level, parentBox.lower.x,
						parentBox.lower.y, parentBox.lower.z, parentBox.upper.x, parentBox.upper.y, parentBox.upper.z);
		const auto path = (this->storageLocation_ / fileName).string();

		const auto [min, max] = searchMinMaxBounds(data);

		const auto generatedNanoVdb = generateNanoVdb(size, maskedValue, 0.0f, data);
		nanovdb::io::writeGrid(path, generatedNanoVdb,
							   nanovdb::io::Codec::NONE); // TODO: enable nanovdb::io::Codec::BLOSC

		totalClusterMemorySize_ += generatedNanoVdb.size();

		result.node.nanoVdbFile = fileName;
		result.node.nanoVdbBufferSize = generatedNanoVdb.size();
		result.node.aabb.min = { static_cast<float>(parentBox.lower.x), static_cast<float>(parentBox.lower.y),
								 static_cast<float>(parentBox.lower.z) };
		result.node.aabb.max = { static_cast<float>(parentBox.upper.x), static_cast<float>(parentBox.upper.y),
								 static_cast<float>(parentBox.upper.z) };
		result.node.minValue = min;
		result.node.maxValue = max;
		result.data = data;
		return result;
	}
};


template <typename DownsamplerType, int MaxVoxelsPerSplitThreshold>
auto BinaryPartitionClusterProcessor<DownsamplerType, MaxVoxelsPerSplitThreshold>::process(const ClusterId clusterId,
																								  const Box3I& rootBox)
	-> ProcessorResult
{
	const auto result = processInternal(clusterId, rootBox);
	return ProcessorResult{ clusterId, totalClusterMemorySize_, result.node };
}
