#pragma once
#include <filesystem>
#include <vector>

#include "SharedStructs.h"

namespace b3d::renderer::nano
{

	enum class RuntimeVolumeState
	{
		loadingRequested,
		ready,
		unloadedRequested,
		unloaded
	};

	struct RuntimeVolume
	{
		NanoVdbVolume volume{};
		RuntimeVolumeState state{};
		owl::AffineSpace3f renormalizeScale{};
	};

	struct VolumeStatistics
	{
		std::map<float, int> histogram;
		int totalValues;
		float min;
		float max;
		float average;
		float median;
	};

	class RuntimeDataSet final
	{
	public:
		RuntimeDataSet();
		auto select(std::size_t index) -> void;
		auto getSelectedData() -> RuntimeVolume&;
		auto addNanoVdb(const std::filesystem::path& path) -> void;
		auto addNanoVdb(const NanoVdbVolume& volume, const VolumeStatistics& statistics) -> void;
		[[nodiscard]] auto getValideVolumeIndicies() const -> std::vector<size_t>;
		[[nodiscard]] auto getStatistics(std::size_t index) const -> const VolumeStatistics& {return volumeStatistics_[index];} 

		RuntimeDataSet(RuntimeDataSet&) = delete;
		RuntimeDataSet(RuntimeDataSet&&) = default;
		~RuntimeDataSet();
	private:
		std::vector<std::filesystem::path> loadableVolumes_{};
		std::vector<RuntimeVolume> runtimeVolumes_{};
		std::vector<VolumeStatistics> volumeStatistics_{};

		VolumeStatistics dummyVolumeStatistics_{};

		std::size_t activeVolume_{ 0 };
		RuntimeVolume dummyVolume_{};
	};

}
