#include "TextureFillerSyncSample.h"

#include <cuda_runtime.h>

#include <cuda/std/cmath>
#include <device_launch_parameters.h>

#include <math_functions.h>

__global__ auto writeVertexBuffer(cudaSurfaceObject_t surface, unsigned int width, unsigned int height, int frmidx) -> void
{
	// gridDim: This variable is of type dim3 and contains the dimensions of the grid.
	// blockIdx: This variable is of type uint3 and contains the block index within the grid.
	// blockDim: This variable is of type dim3 and contains the dimensions of the block.
	// threadIdx: This variable is of type uint3 and contains the thread index within the block.

	const auto x = min(blockIdx.x * blockDim.x + threadIdx.x, width - 1);
	const auto y = min(blockIdx.y * blockDim.y + threadIdx.y, height - 1);

	auto col1 = uint32_t{ 0xFFFFFFFF };
	auto col2 = uint32_t{ 0xFF0000FF };

	float time = (float)frmidx / 1000.0f;
	float val1 = (sin(time) + 1.0f) * 0.5f;

	float mixedVal = col1 * val1 + col2 * (1.0f - val1);

	if (x + y == 0)
	{
		printf("Hello from global thread 0");
	}
	surf2Dwrite(mixedVal, surface, x * sizeof(uint32_t), y);
}

__global__ void kernel()
{
#if __CUDA_ARCH__ >= 700
	for (int i = 0; i < 1000; i++)
		//__nanosleep(1000000U); // ls
		__nanosleep(1000U); // ls
#else
	printf(">>> __CUDA_ARCH__ !\n");
#endif
}

void b3d::renderer::TextureFillerSyncSample::onRender()
{
	static int idx = 0;
	
	idx++;
	auto renderTargetFeatureParams = renderTargetFeature_->getParamsData();

	const auto fbSize = dim3{
		renderTargetFeatureParams.colorRT.extent.width,
		renderTargetFeatureParams.colorRT.extent.height,
	};

	auto cudaRet = cudaSuccess;
	// Execute Kernel
	{
		const auto gridDimXAdd = fbSize.x % 2 == 0 ? 0 : 1;
		const auto gridDimYAdd = fbSize.y % 2 == 0 ? 0 : 1;
		auto gridDim = dim3{ fbSize.x / 32 + gridDimXAdd, fbSize.y / 32 + gridDimYAdd };
		auto blockDim = dim3{ 32, 32 };
		writeVertexBuffer<<<gridDim, blockDim>>>(renderTargetFeatureParams.colorRT.surfaces[0].surface, fbSize.x, fbSize.y, idx);
		kernel<<<1, 1>>>();

		// cudaRet = cudaGetLastError();
	}
}


auto b3d::renderer::TextureFillerSyncSample::onInitialize() -> void
{
}

auto b3d::renderer::TextureFillerSyncSample::onDeinitialize() -> void
{
}

auto b3d::renderer::TextureFillerSyncSample::onGui() -> void
{
}
