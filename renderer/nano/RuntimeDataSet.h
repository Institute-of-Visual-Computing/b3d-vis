#pragma once
#include <filesystem>
#include <vector>


#include <SharedStructs.h>

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

	class RuntimeDataSet
	{
	public:
		RuntimeDataSet();
		auto select(std::size_t index) -> void;
		auto getSelectedData() -> RuntimeVolume&;

	private:
		std::vector<std::filesystem::path> loadableVolumes_{};
		std::vector<RuntimeVolume> runtimeVolumes_{};
		std::size_t activeVolume_{ 0 };
	};

}
