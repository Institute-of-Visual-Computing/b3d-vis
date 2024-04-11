#include "RuntimeDataSet.h"

#include <filesystem>

#include "NanoVDB.h"
#include "owl/common/math/AffineSpace.h"
#include "owl/helper/cuda.h"
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

	auto createVolumeFromFile(const std::filesystem::path& file) -> NanoVdbVolume
	{
		//TODO: Let's use shared parameters to grab an initial volume path from the viewer
		// const auto testFile = std::filesystem::path{ "D:/datacubes/n4565_cut/funny.nvdb" };
		// const auto testFile =
		std::filesystem::path{ "D:/datacubes/n4565_cut/filtered_level_0_224_257_177_id_7_upscale.fits.nvdb" };
		//std::filesystem::path{ "C:/Users/anton/Downloads/chameleon_1024x1024x1080_uint16.nvdb" };
		//std::filesystem::path{ "C:/Users/anton/Downloads/carp_256x256x512_uint16.nvdb" };
		const auto testFile = std::filesystem::path{ "D:/datacubes/n4565_cut/nano_level_0_224_257_177.nvdb" };
		// const auto testFile = std::filesystem::path{ "D:/datacubes/ska/40gb/sky_ldev_v2.nvdb" };

		assert(std::filesystem::exists(testFile));
		const auto gridVolume = nanovdb::io::readGrid(testFile.string());
		return createVolume(gridVolume);
	}
}

b3d::renderer::nano::RuntimeDataSet::RuntimeDataSet()
{
	const auto volume = createVolume(nanovdb::createFogVolumeTorus());
	const auto volumeSize = volume.indexBox.size();
	const auto longestAxis = std::max({ volumeSize.x, volumeSize.y, volumeSize.z });

	const auto scale = 1.0f / longestAxis;

	const auto renormalizeScale = owl::AffineSpace3f::scale(owl::vec3f{ scale, scale, scale });

	runtimeVolumes_.push_back(
		RuntimeVolume{ volume, RuntimeVolumeState::ready, renormalizeScale });
}
auto RuntimeDataSet::select(std::size_t index) -> void
{
	assert(index < runtimeVolumes_.size());
	activeVolume_ = index;
}
auto RuntimeDataSet::getSelectedData() -> RuntimeVolume&
{
	return runtimeVolumes_[activeVolume_];
}
