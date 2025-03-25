#pragma once
#include <filesystem>
#include <vector>

#include <cuda_runtime.h>
#include <mutex>

namespace b3d::tools::renderer::nvdb
{
	enum class RuntimeVolumeState;
	struct RuntimeVolume;

	class RuntimeDataset final
	{
	public:
		RuntimeDataset();
		auto getVolumeState(const std::string& uuid) -> std::optional<RuntimeVolumeState>;

		auto addNanoVdb(const std::filesystem::path& path, cudaStream_t stream = 0, const std::string& volumeUuid = "")
			-> void;

		auto getRuntimeVolume(const std::string& uuid) -> std::optional<RuntimeVolume>;

		RuntimeDataset(RuntimeDataset&) = delete;
		RuntimeDataset(RuntimeDataset&&) = default;
		~RuntimeDataset();

		// ############### Legacy
		auto select(std::size_t index) -> void;
		auto getSelectedData() -> RuntimeVolume;

		// ############### Legacy

	private:
		std::vector<RuntimeVolume> runtimeVolumes_;

		std::mutex listMutex_;

		// ############### Legacy
		std::size_t activeVolume_{ 0 };
		std::unique_ptr<RuntimeVolume> dummyVolume_{};
		// ############### Legacy

	};

} // namespace b3d::tools::renderer::nano
