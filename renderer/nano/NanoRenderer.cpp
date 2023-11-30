#include "NanoRenderer.h"
#include <Logging.h>
#include <nanovdb/NanoVDB.h>
#include <owl/helper/cuda.h>
#include "cuda_runtime.h"

#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/OpenToNanoVDB.h>
#include <nanovdb/util/Primitives.h>

#include <cuda.h>



using namespace b3d::renderer;

struct NanoVDBVolume
{
	owl::box3f indexBox;
	owl::box3f worldAABB;
	owl::LinearSpace3f transform;
	CUdeviceptr volume = 0;
};

namespace
{
	auto foo() -> void
	{
		auto volume = NanoVDBVolume{};
		const auto gridVolume = nanovdb::createLevelSetSphere<float, float>(10.0f);
		OWL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&volume.volume), gridVolume.size()));
		OWL_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(volume.volume), gridVolume.data(), gridVolume.size(),
								  cudaMemcpyHostToDevice));

		const auto gridHandle = gridVolume.grid<float>();
		const auto& map = gridHandle->map();
		volume.transform = owl::LinearSpace3f{ map.mMatF[0], map.mMatF[1], map.mMatF[2], map.mMatF[3], map.mMatF[4],
											   map.mMatF[5], map.mMatF[6], map.mMatF[7], map.mMatF[8] };

		{
			const auto& box = gridVolume.gridMetaData()->worldBBox();
			const auto min = owl::vec3f{ static_cast<float>(box.min()[0]), static_cast<float>(box.min()[1]),
										 static_cast<float>(box.min()[2]) };
			const auto max = owl::vec3f{ static_cast<float>(box.max()[0]), static_cast<float>(box.max()[1]),
										 static_cast<float>(box.max()[2]) };

			volume.worldAABB = owl::box3f{ min, max };
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
	}
} // namespace

auto NanoRenderer::onRender(const View& view) -> void
{

	auto waitParams = cudaExternalSemaphoreWaitParams{};
	waitParams.flags = 0;
	waitParams.params.fence.value = 0;
	cudaWaitExternalSemaphoresAsync(&initializationInfo_.signalSemaphore, &waitParams, 1);

	auto signalParams = cudaExternalSemaphoreSignalParams{};
	signalParams.flags = 0;
	signalParams.params.fence.value = 0;
	cudaSignalExternalSemaphoresAsync(&initializationInfo_.waitSemaphore, &signalParams, 1);
}

auto NanoRenderer::onInitialize() -> void
{
	foo();
	log("[NanoRenderer] onInitialize!");
}

auto NanoRenderer::onDeinitialize() -> void
{
	log("[NanoRenderer] onDeinitialize!");
}

auto NanoRenderer::onGui() -> void
{
}
