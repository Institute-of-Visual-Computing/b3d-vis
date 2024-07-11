#include "RuntimeDataSet.h"

#include <filesystem>
#include <concepts>

#include <execution>
#include "NanoVDB.h"
#include "owl/common/math/AffineSpace.h"
#include "owl/helper/cuda.h"
#include "util/GridStats.h"
#include "util/IO.h"
#include "util/Primitives.h"

using namespace b3d::renderer::nano;

namespace
{
	auto createVolume(const nanovdb::GridHandle<>& gridVolume) -> NanoVdbVolume
	{
		auto volume = NanoVdbVolume{};


		OWL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&volume.grid), gridVolume.size()));
		OWL_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(volume.grid), gridVolume.data(), gridVolume.size(),
								  cudaMemcpyHostToDevice));

		const auto gridHandle = gridVolume.grid<float>();
		const auto& map = gridHandle->mMap;
		const auto orientation =
			owl::LinearSpace3f{ map.mMatF[0], map.mMatF[1], map.mMatF[2], map.mMatF[3], map.mMatF[4],
								map.mMatF[5], map.mMatF[6], map.mMatF[7], map.mMatF[8] };
		const auto position = owl::vec3f{ 0.0, 0.0, 0.0 };

		volume.transform = owl::AffineSpace3f{ orientation, position };

		{
			const auto& box = gridVolume.gridMetaData()->worldBBox();
			const auto min = owl::vec3f{ static_cast<float>(box.min()[0]), static_cast<float>(box.min()[1]),
										 static_cast<float>(box.min()[2]) };
			const auto max = owl::vec3f{ static_cast<float>(box.max()[0]), static_cast<float>(box.max()[1]),
										 static_cast<float>(box.max()[2]) };
			volume.worldAabb = owl::box3f{ min, max };
		}

		{
			const auto indexBox = gridHandle->indexBBox();
			const auto boundsMin = nanovdb::Coord{ indexBox.min() };
			const auto boundsMax = nanovdb::Coord{ indexBox.max() + nanovdb::Coord(1) };

			const auto min = owl::vec3f{ static_cast<float>(boundsMin[0]), static_cast<float>(boundsMin[1]),
										 static_cast<float>(boundsMin[2]) };
			const auto max = owl::vec3f{ static_cast<float>(boundsMax[0]), static_cast<float>(boundsMax[1]),
										 static_cast<float>(boundsMax[2]) };

			volume.indexBox = owl::box3f{ min, max };
		}

		return volume;
	}

	auto computeStatistics(const nanovdb::GridHandle<>& gridVolume) -> VolumeStatistics
	{
		const auto gridHandle = gridVolume.grid<float>();
		const auto indexBox = gridHandle->indexBBox();
		auto ac = gridHandle->getAccessor();
		auto histogram = std::map<float, int>();
		auto min = std::numeric_limits<float>::max();
		auto max = std::numeric_limits<float>::min();
		auto totalSamples = 0;
		auto sum = 0.0;

		auto addValueToStatistics =
			[&](const nanovdb::Coord& coord)
		{
			auto leaf = ac.probeLeaf(coord);
			if (leaf == nullptr)
			{
				return;
			}
			auto leafData = leaf->data();
			auto value = leafData->getAvg();
			if (!histogram.contains(value))
			{
				histogram[value] = 1;
			}
			histogram[value]++;
			totalSamples++;

			if (value < min)
			{
				min = value;
			}

			if (value > max)
			{
				max = value;
			}
			sum += value;
		};

		std::vector<nanovdb::Coord> ii;
		std::cout << indexBox.volume() << std::endl;
		ii.reserve(indexBox.volume());
		for (auto i = indexBox.begin(); i != indexBox.end(); i++)
		{
			ii.push_back(*i);
		}

		std::for_each(std::execution::par,  ii.begin(), ii.end(), addValueToStatistics);


		/*for (auto i = indexBox.begin(); i != indexBox.end(); i++)
		{
			addValueToStatistics(*i);
		}*/

		const auto average = (float)(sum / totalSamples);

		auto halfValue = totalSamples / 2;

		auto median = 0.0f;

		auto samplesCount = 0;


		for (auto& [key, value] : histogram)
		{
			if (samplesCount > halfValue)
			{
				median = key;
				break;
			}
			samplesCount += value;
		}

		return VolumeStatistics{ .histogram = histogram,
								 .totalValues = totalSamples,
								 .min = min,
								 .max = max,
								 .average = average,
								 .median = median };
	}

	auto createVolumeFromFile(const std::filesystem::path& file) -> NanoVdbVolume
	{
		// TODO: Let's use shared parameters to grab an initial volume path from the viewer
		//  const auto testFile = std::filesystem::path{ "D:/datacubes/n4565_cut/funny.nvdb" };
		//  const auto testFile =
		// std::filesystem::path{ "D:/datacubes/n4565_cut/filtered_level_0_224_257_177_id_7_upscale.fits.nvdb" };
		// std::filesystem::path{ "C:/Users/anton/Downloads/chameleon_1024x1024x1080_uint16.nvdb" };
		// std::filesystem::path{ "C:/Users/anton/Downloads/carp_256x256x512_uint16.nvdb" };
		// const auto testFile = std::filesystem::path{ "D:/datacubes/n4565_cut/nano_level_0_224_257_177.nvdb" };
		// const auto testFile = std::filesystem::path{ "D:/datacubes/ska/40gb/sky_ldev_v2.nvdb" };

		assert(std::filesystem::exists(file));
		const auto gridVolume = nanovdb::io::readGrid(file.string());
		return createVolume(gridVolume);
	}
} // namespace

b3d::renderer::nano::RuntimeDataSet::RuntimeDataSet()
{
	// TODO: use cudaMemGetInfo(), add LRU eviction strategy, pass data pool size via parameter

	// addNanoVdb(createVolume(nanovdb::createFogVolumeTorus()));
	auto gridHandle = nanovdb::createFogVolumeSphere();
	const auto volume = createVolume(gridHandle);
	const auto volumeSize = volume.indexBox.size();
	const auto longestAxis = std::max({ volumeSize.x, volumeSize.y, volumeSize.z });

	const auto scale = 1.0f / longestAxis;

	const auto renormalizeScale = owl::AffineSpace3f::scale(owl::vec3f{ scale, scale, scale });
	dummyVolume_ = RuntimeVolume{ volume, RuntimeVolumeState::ready, renormalizeScale };
	dummyVolumeStatistics_ = computeStatistics(gridHandle);
}
auto RuntimeDataSet::select(const std::size_t index) -> void
{
	assert(index <= runtimeVolumes_.size());
	activeVolume_ = index;
}
auto RuntimeDataSet::getSelectedData() -> RuntimeVolume&
{
	if (runtimeVolumes_.empty())
	{
		return dummyVolume_;
	}
	return runtimeVolumes_[activeVolume_];
}
auto RuntimeDataSet::addNanoVdb(const std::filesystem::path& path) -> void
{
	assert(std::filesystem::exists(path));
	const auto gridVolume = nanovdb::io::readGrid(path.string());
	addNanoVdb(createVolume(gridVolume), computeStatistics(gridVolume));
}
auto RuntimeDataSet::addNanoVdb(const NanoVdbVolume& volume, const VolumeStatistics& statistics) -> void
{
	const auto volumeSize = volume.indexBox.size();
	const auto longestAxis = std::max({ volumeSize.x, volumeSize.y, volumeSize.z });

	const auto scale = 1.0f / longestAxis;

	const auto renormalizeScale = owl::AffineSpace3f::scale(owl::vec3f{ scale, scale, scale });

	runtimeVolumes_.push_back(RuntimeVolume{ volume, RuntimeVolumeState::ready, renormalizeScale });
	volumeStatistics_.push_back(statistics);
}
RuntimeDataSet::~RuntimeDataSet()
{
	for (auto& volume : runtimeVolumes_)
	{
		OWL_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(volume.volume.grid)));
		volume.volume.grid = {};
	}
}

auto RuntimeDataSet::getValideVolumeIndicies() const -> std::vector<size_t>
{
	auto indicies = std::vector<size_t>{};

	for (auto i = 0u; i < runtimeVolumes_.size(); i++)
	{
		const auto& runtimeVolume = runtimeVolumes_[i];
		if (runtimeVolume.state == RuntimeVolumeState::ready)
		{
			indicies.push_back(i);
		}
	}
	return indicies;
}
