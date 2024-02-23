#include "CudaSurfaceObjectWriteTestRenderer.h"

#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include "cuda/std/cmath"

#include "math_functions.h"

using namespace b3d::renderer;

__global__ auto writeVertexBuffer(cudaSurfaceObject_t surface, unsigned int width, unsigned int height) -> void
{
	// gridDim: This variable is of type dim3 and contains the dimensions of the grid.
	// blockIdx: This variable is of type uint3 and contains the block index within the grid.
	// blockDim: This variable is of type dim3 and contains the dimensions of the block.
	// threadIdx: This variable is of type uint3 and contains the thread index within the block.

	const auto x = min(blockIdx.x * blockDim.x + threadIdx.x, width - 1);
	const auto y = min(blockIdx.y * blockDim.y + threadIdx.y, height - 1);

	auto val = uint32_t{ 0xFFFFFFFF };
	
	if (x + y == 0)
	{
		// printf("Hello from global thread 0\n");
	}
	surf2Dwrite(val, surface, x * sizeof(uint32_t), y);
}


auto CudaSurfaceObjectWriteTestRenderer::onRender() -> void
{
	// TODO: class members
	std::array<cudaArray_t, 2> cudaArrays{};
	std::array<cudaSurfaceObject_t, 2> cudaSurfaceObjects{};

	const auto renderTargets = renderData_->get<RenderTargets>("renderTargets");

	auto cudaRet = cudaSuccess;
	// Map and createSurface
	{
		cudaRet = cudaGraphicsMapResources(1, const_cast<cudaGraphicsResource_t*>(&renderTargets->colorRt.target));
		for (auto i = 0; i < renderTargets->colorRt.extent.depth; i++)
		{
			cudaRet = cudaGraphicsSubResourceGetMappedArray(&cudaArrays[i], renderTargets->colorRt.target, i, 0);

			cudaResourceDesc resDesc{};
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cudaArrays[i];
			cudaRet = cudaCreateSurfaceObject(&cudaSurfaceObjects[i], &resDesc);
		}
	}

	// Execute Kernel
	{
		const auto gridDimXAdd = renderTargets->colorRt.extent.width % 32 == 0 ? 0 : 1; 
		const auto gridDimYAdd = renderTargets->colorRt.extent.height % 32 == 0 ? 0 : 1; 
		auto gridDim = dim3{ renderTargets->colorRt.extent.width / 32 + gridDimXAdd,
							 renderTargets->colorRt.extent.height / 32 + gridDimYAdd };
		auto blockDim = dim3{ 32, 32 };
		writeVertexBuffer<<<gridDim, blockDim>>>(cudaSurfaceObjects[0], renderTargets->colorRt.extent.width,
												 renderTargets->colorRt.extent.height);
		cudaDeviceSynchronize();
		cudaRet = cudaGetLastError();
	}

	// test Copy - Uncomment to test
	if constexpr (false)
	{
		std::vector<uint32_t> hostMem;
		hostMem.resize(renderTargets->colorRt.extent.width * renderTargets->colorRt.extent.height);
		cudaMemcpy2DFromArray(hostMem.data(), renderTargets->colorRt.extent.width * sizeof(uint32_t), cudaArrays[0], 0,
							  0, renderTargets->colorRt.extent.width * sizeof(uint32_t),
							  renderTargets->colorRt.extent.height,
							  cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		cudaDeviceSynchronize();
	}

	// Destroy and unmap
	{
		for (auto i = 0; i < renderTargets->colorRt.extent.depth; i++)
		{
			cudaRet = cudaDestroySurfaceObject(cudaSurfaceObjects[i]);
		}
		cudaRet = cudaGraphicsUnmapResources(1, const_cast<cudaGraphicsResource_t*>(&renderTargets->colorRt.target));
	}
}

auto CudaSurfaceObjectWriteTestRenderer::onInitialize() -> void
{
	
}

auto CudaSurfaceObjectWriteTestRenderer::onDeinitialize() -> void
{
	
}

auto CudaSurfaceObjectWriteTestRenderer::onGui() -> void
{
	
}
