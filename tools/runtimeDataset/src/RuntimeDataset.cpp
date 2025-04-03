#include "RuntimeDataset.h"

#include <concepts>
#include <filesystem>
#include <execution>

#include <owl/common/math/AffineSpace.h>
#include <owl/helper/cuda.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/CreatePrimitives.h>
#include <nanovdb/tools/GridStats.h>

#include <SharedRenderingStructs.h>

using namespace b3d::tools::renderer::nvdb;

namespace
{
	auto createVolume(const nanovdb::GridHandle<>& gridVolume, cudaStream_t stream)
		-> b3d::tools::renderer::nvdb::NanoVdbVolume
	{
		auto volume = b3d::tools::renderer::nvdb::NanoVdbVolume{};

		std::cout << "starting transfer\n";
		OWL_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&volume.grid), gridVolume.size(), stream));
		OWL_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(volume.grid), gridVolume.data(), gridVolume.size(),
									   cudaMemcpyHostToDevice, stream));
		cudaStreamSynchronize(stream);
		std::cout << "finishing transfer\n";

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

	auto setRuntimeVolumeReady(void* runtimeVolumePointer) -> void
	{
		static_cast<RuntimeVolume*>(runtimeVolumePointer)->state = RuntimeVolumeState::ready;
	}

	auto loadVolumeToDevice(RuntimeVolume& container, const nanovdb::GridHandle<>& gridVolume, cudaStream_t stream)
		-> void
	{
		std::cout << "starting transfer\n";
		OWL_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&container.volume.grid), gridVolume.size(), stream));
		OWL_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(container.volume.grid), gridVolume.data(),
									   gridVolume.size(), cudaMemcpyHostToDevice, stream));
		// TODO:
		OWL_CUDA_CHECK(cudaLaunchHostFunc(stream, &setRuntimeVolumeReady, &container));

		const auto gridHandle = gridVolume.grid<float>();
		const auto& map = gridHandle->mMap;
		const auto orientation =
			owl::LinearSpace3f{ map.mMatF[0], map.mMatF[1], map.mMatF[2], map.mMatF[3], map.mMatF[4],
								map.mMatF[5], map.mMatF[6], map.mMatF[7], map.mMatF[8] };
		const auto position = owl::vec3f{ 0.0, 0.0, 0.0 };

		container.volume.transform = owl::AffineSpace3f{ orientation, position };

		{
			const auto& box = gridVolume.gridMetaData()->worldBBox();
			const auto min = owl::vec3f{ static_cast<float>(box.min()[0]), static_cast<float>(box.min()[1]),
										 static_cast<float>(box.min()[2]) };
			const auto max = owl::vec3f{ static_cast<float>(box.max()[0]), static_cast<float>(box.max()[1]),
										 static_cast<float>(box.max()[2]) };
			container.volume.worldAabb = owl::box3f{ min, max };
		}

		{
			const auto indexBox = gridHandle->indexBBox();
			const auto boundsMin = nanovdb::Coord{ indexBox.min() };
			const auto boundsMax = nanovdb::Coord{ indexBox.max() + nanovdb::Coord(1) };

			const auto min = owl::vec3f{ static_cast<float>(boundsMin[0]), static_cast<float>(boundsMin[1]),
										 static_cast<float>(boundsMin[2]) };
			const auto max = owl::vec3f{ static_cast<float>(boundsMax[0]), static_cast<float>(boundsMax[1]),
										 static_cast<float>(boundsMax[2]) };

			container.volume.indexBox = owl::box3f{ min, max };
		}
	}

} // namespace

b3d::tools::renderer::nvdb::RuntimeDataset::RuntimeDataset()
{
	runtimeVolumes_ = std::vector<RuntimeVolume>();
	// TODO: use cudaMemGetInfo(), add LRU eviction strategy, pass data pool size via parameter

	// addNanoVdb(createVolume(nanovdb::createFogVolumeTorus()));
	auto gridHandle = nanovdb::tools::createFogVolumeSphere();
	const auto volume = createVolume(gridHandle, 0);
	const auto volumeSize = volume.indexBox.size();
	const auto longestAxis = std::max({ volumeSize.x, volumeSize.y, volumeSize.z });

	const auto scale = 1.0f / longestAxis;

	const auto renormalizeScale = owl::AffineSpace3f::scale(owl::vec3f{ scale, scale, scale });
	dummyVolume_ = std::make_unique<RuntimeVolume>(volume, RuntimeVolumeState::ready, renormalizeScale);
}

auto RuntimeDataset::getVolumeState(const std::string& uuid) -> std::optional<RuntimeVolumeState>
{
	const auto found = std::ranges::find_if(runtimeVolumes_, [&](const auto& volume) { return volume.uuid == uuid; });
	if (found != runtimeVolumes_.end())
	{
		return found->state;
	}
	return std::nullopt;
}

auto b3d::tools::renderer::nvdb::RuntimeDataset::addNanoVdb(const std::filesystem::path& path, cudaStream_t stream,
															const std::string& volumeUuid) -> void
{
	assert(std::filesystem::exists(path));
	listMutex_.lock();
	if (std::ranges::find_if(runtimeVolumes_, [&](const auto& volume) { return volume.uuid == volumeUuid; }) !=
		runtimeVolumes_.end())
	{
		listMutex_.unlock();
		return;
	}
	runtimeVolumes_.push_back({ .uuid = volumeUuid });
	auto& volume = runtimeVolumes_.back();
	listMutex_.unlock();

	const auto gridVolume = nanovdb::io::readGrid(path.string());

	loadVolumeToDevice(volume, gridVolume, stream);
	const auto volumeSize = volume.volume.indexBox.size();
	const auto longestAxis = std::max({ volumeSize.x, volumeSize.y, volumeSize.z });
	const auto scale = 1.0f / longestAxis;
	const auto renormalizeScale = owl::AffineSpace3f::scale(owl::vec3f{ scale, scale, scale });
	volume.renormalizeScale = renormalizeScale;
}

auto RuntimeDataset::getRuntimeVolume(const std::string& uuid) -> std::optional<RuntimeVolume>
{
	const auto found = std::ranges::find_if(runtimeVolumes_, [&](const auto& volume) { return volume.uuid == uuid; });
	if (found != runtimeVolumes_.end())
	{
		return *found;
	}
	return std::nullopt;
}

RuntimeDataset::~RuntimeDataset()
{
	for (auto& volume : runtimeVolumes_)
	{
		OWL_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(volume.volume.grid)));
		volume.volume.grid = {};
	}
}

auto RuntimeDataset::select(std::size_t index) -> void
{
	const std::lock_guard listGuard(listMutex_);
	assert(index <= runtimeVolumes_.size());
	activeVolume_ = index;
}

auto RuntimeDataset::getSelectedData() -> RuntimeVolume
{
	const std::lock_guard listGuard(listMutex_);
	if (runtimeVolumes_.empty())
	{
		return *dummyVolume_;
	}
	return runtimeVolumes_[activeVolume_];
}
