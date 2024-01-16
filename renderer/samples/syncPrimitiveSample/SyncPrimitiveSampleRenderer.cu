#include "SyncPrimitiveSampleRenderer.h"

#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include "cuda/std/cmath"

#include "math_functions.h"

__global__ auto writeVertexBuffer(cudaSurfaceObject_t surface, unsigned int width, unsigned int height) -> void
{
	// gridDim: This variable is of type dim3 and contains the dimensions of the grid.
	// blockIdx: This variable is of type uint3 and contains the block index within the grid.
	// blockDim: This variable is of type dim3 and contains the dimensions of the block.
	// threadIdx: This variable is of type uint3 and contains the thread index within the block.

	const auto x = min(blockIdx.x * blockDim.x + threadIdx.x, width - 1);
	const auto y = min(blockIdx.y * blockDim.y + threadIdx.y, height - 1);

	auto val = uint32_t{ 0xFFFFFFFF };
	val = uint32_t{ 0xFF0000FF };
	if (x + y == 0)
	{
		printf("Hello from global thread 0");
	}
	surf2Dwrite(val, surface, x * sizeof(uint32_t), y);
}

__global__ void kernel()
{
#if __CUDA_ARCH__ >= 700
	for (int i = 0; i < 1000; i++)
		__nanosleep(1000000U); // ls
#else
	printf(">>> __CUDA_ARCH__ !\n");
#endif
}

void b3d::renderer::SyncPrimitiveSampleRenderer::onRender(const View& view)
{

	auto cudaRet = cudaSuccess;

	auto waitParams = cudaExternalSemaphoreWaitParams{};
	waitParams.flags = 0;
	waitParams.params.fence.value = view.fenceValue;
	cudaRet = cudaWaitExternalSemaphoresAsync(&initializationInfo_.signalSemaphore, &waitParams, 1);

	// TODO: class members
	std::array<cudaArray_t, 2> cudaArrays{};
	std::array<cudaSurfaceObject_t, 2> cudaSurfaceObjects{};

	// Map and createSurface
	{
		cudaRet = cudaGraphicsMapResources(1, const_cast<cudaGraphicsResource_t*>(&view.colorRt.target));
		for (auto i = 0; i < view.colorRt.extent.depth; i++)
		{
			cudaRet = cudaGraphicsSubResourceGetMappedArray(&cudaArrays[i], view.colorRt.target, i, 0);
			 
			cudaResourceDesc resDesc{};
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cudaArrays[i];
			cudaRet = cudaCreateSurfaceObject(&cudaSurfaceObjects[i], &resDesc);
		}
	}

	// Execute Kernel
	{
		const auto gridDimXAdd = view.colorRt.extent.width % 2 == 0 ? 0 : 1;
		const auto gridDimYAdd = view.colorRt.extent.height % 2 == 0 ? 0 : 1;
		auto gridDim =
			dim3{ view.colorRt.extent.width / 32 + gridDimXAdd, view.colorRt.extent.height / 32 + gridDimYAdd };
		auto blockDim = dim3{ 32, 32 };
		writeVertexBuffer<<<gridDim, blockDim>>>(cudaSurfaceObjects[0], view.colorRt.extent.width, view.colorRt.extent.height);
		kernel<<<1, 1>>>();
		
		//cudaRet = cudaGetLastError();
	}

	// test Copy - Uncomment to test
	if constexpr (false)
	{
		std::vector<uint32_t> hostMem;
		hostMem.resize(view.colorRt.extent.width * view.colorRt.extent.height);
		cudaMemcpy2DFromArray(hostMem.data(), view.colorRt.extent.width * sizeof(uint32_t), cudaArrays[0], 0, 0,
							  view.colorRt.extent.width * sizeof(uint32_t), view.colorRt.extent.height,
							  cudaMemcpyDeviceToHost);
	}

	// Destroy and unmap
	{
		for (auto i = 0; i < view.colorRt.extent.depth; i++)
		{
			cudaRet = cudaDestroySurfaceObject(cudaSurfaceObjects[i]);
		}
		cudaRet = cudaGraphicsUnmapResources(1, const_cast<cudaGraphicsResource_t*>(&view.colorRt.target));
	}

	auto signalParams = cudaExternalSemaphoreSignalParams{};
	signalParams.flags = 0;
	signalParams.params.fence.value = view.fenceValue;
	cudaRet = cudaSignalExternalSemaphoresAsync(&initializationInfo_.waitSemaphore, &signalParams, 1);

	cudaError_t rc = cudaGetLastError();
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "error (%s: line %d): %d:  %s\n", __FILE__, __LINE__, rc, cudaGetErrorString(rc));
		// OWL_RAISE("fatal cuda error");
	}
}

auto b3d::renderer::SyncPrimitiveSampleRenderer::onInitialize() -> void
{

}

auto b3d::renderer::SyncPrimitiveSampleRenderer::onDeinitialize() -> void
{
}

auto b3d::renderer::SyncPrimitiveSampleRenderer::onGui() -> void
{
}
