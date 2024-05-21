#include "FoveatedRendering.h"

#include <owl/common.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "owl/helper/cuda.h"

// #include "FoveatedHelper.cuh"
#include <cuda/std/cmath>


#include "FoveatedHelper.cuh"
#include "OptixHelper.cuh"
#include "imgui.h"
#include "nanovdb/NanoVDB.h"
#include "owl/owl_device.h"

using namespace b3d::renderer;
using namespace owl;

struct FovealParameter
{
	vec2f fovealScreenSpace;
	float scaleRatio;
	float kernelParameter;
};

__global__ auto resolveLp(cudaTextureObject_t inputTextureObj, cudaSurfaceObject_t outputSurfObj, const int width,
	const int height, const FovealParameter fovealParameter)
	-> void
{
	const auto foveal = fovealParameter.fovealScreenSpace;
	const auto resolution = vec2f(width, height);// / fovealParameter.scaleRatio;

	const auto x = blockIdx.x * blockDim.x + threadIdx.x;
	const auto y = blockIdx.y * blockDim.y + threadIdx.y;
	const auto pixelIndex = vec2f(x, y);
	if (pixelIndex.x >= width || pixelIndex.y >= height)
	{
		return;
	}
	auto newPixelIndex = vec2f(pixelIndex);

	auto screen = (vec2f(newPixelIndex));// + vec2f(.5f);

	screen = inverseLogMap(fovealParameter.scaleRatio, screen, foveal, resolution);
	screen /= resolution;

	const auto result = tex2D<float4>(inputTextureObj, screen.x, screen.y);

	surf2Dwrite(owl::make_rgba(vec4f(result.x, result.y, result.z, result.w)), outputSurfObj, sizeof(uint32_t) * pixelIndex.x, pixelIndex.y);
}

auto b3d::renderer::FoveatedRenderingFeature::onInitialize() -> void
{
	createResources();
	const auto foveatedControlData = sharedParameters_->get<FoveatedRenderingControl>("foveatedRenderingControl");
	if (!foveatedControlData)
	{
		throw std::runtime_error("Missing shared parameters for foveated rendering feature!");
	}
	controlData_ = foveatedControlData;
	resolutionScaleRatio_ = controlData_->temporalBufferResolutionRelativeScale;
}
auto b3d::renderer::FoveatedRenderingFeature::onDeinitialize() -> void
{
	destroyResources();
}
auto b3d::renderer::FoveatedRenderingFeature::gui() -> void
{
	ImGui::Checkbox("Enable Feature", &controlData_->isEnabled);
	ImGui::BeginDisabled(! controlData_->isEnabled);
	ImGui::TextWrapped("Hold SPACE to move both foveal points with mouse. Hold LEFT ARROW KEY/RIGHT ARROW KEY to move left/right foveal points with mouse.");
	ImGui::Text("Left Foveal x:%.2f y:%.2f", controlData_->leftEyeGazeScreenSpace.x, controlData_->leftEyeGazeScreenSpace.y);
	ImGui::Text("Right Foveal x:%.2f y:%.2f", controlData_->rightEyeGazeScreenSpace.x, controlData_->rightEyeGazeScreenSpace.y);
	ImGui::Separator();
	ImGui::SliderFloat("Kernel Parameter", &controlData_->kernelParameter, 1.0f, 6.0f);
	ImGui::SliderFloat("LP Buffer Scale", &controlData_->temporalBufferResolutionRelativeScale, 1.0, 4.0f);
	ImGui::SameLine();

	const auto disableApplyButton = resolutionScaleRatio_ == controlData_->temporalBufferResolutionRelativeScale;

	ImGui::BeginDisabled(disableApplyButton);
	if(ImGui::Button("Apply"))
	{
		resolutionScaleRatio_ = controlData_->temporalBufferResolutionRelativeScale;
		OWL_CUDA_CHECK(cudaDeviceSynchronize());
		destroyResources();
		createResources();
	}
	ImGui::EndDisabled();

	ImGui::EndDisabled();
	const auto mousePosition = ImGui::GetMousePos();

	const auto displaySize = ImGui::GetIO().DisplaySize;

	auto mouseScreenSpace = owl::vec2f{};
	mouseScreenSpace.x = mousePosition.x / static_cast<float>(displaySize.x) * 2.0f - 1.0f;
	mouseScreenSpace.y = (1.0 - mousePosition.y / static_cast<float>(displaySize.y)) * 2.0f - 1.0f;

	if (ImGui::GetIO().KeysDown[ImGuiKey_LeftArrow])
	{
		controlData_->leftEyeGazeScreenSpace = mouseScreenSpace;
	}

	if (ImGui::GetIO().KeysDown[ImGuiKey_RightArrow])
	{
		controlData_->rightEyeGazeScreenSpace = mouseScreenSpace;
	}

	if (ImGui::GetIO().KeysDown[ImGuiKey_Space])
	{
		controlData_->leftEyeGazeScreenSpace = mouseScreenSpace;
		controlData_->rightEyeGazeScreenSpace = mouseScreenSpace;

	}
}
auto b3d::renderer::FoveatedRenderingFeature::resolve(const CudaSurfaceResource& surface, const uint32_t width, const uint32_t height, const CUstream stream, float fovX, float fovY) -> void
{
	dim3 threadsPerBlock(16, 16);
	const auto x = (width + threadsPerBlock.x - 1) / threadsPerBlock.x;
	const auto y = (height + threadsPerBlock.y - 1) / threadsPerBlock.y;
	dim3 numBlocks(x, y);

	const auto resolution = vec2f{ static_cast<float>(width), static_cast<float>(height) } / resolutionScaleRatio_;
	const auto fovealParameter = FovealParameter{ (vec2f{fovX,fovY} / 2.0f + 0.5f) * resolution, resolutionScaleRatio_, controlData_->kernelParameter };

	resolveLp << <numBlocks, threadsPerBlock, 0, stream >> > (
		lpResources_.front().texture,
		surface.surface,
		width,
		height,
		{ vec2f{fovX,fovY}, resolutionScaleRatio_, controlData_->kernelParameter });


}
auto b3d::renderer::FoveatedRenderingFeature::destroyResources() -> void
{
	for (const auto& resource : lpResources_)
	{
		OWL_CUDA_CHECK(cudaDestroySurfaceObject(resource.surface.surface));
		OWL_CUDA_CHECK(cudaDestroyTextureObject(resource.texture));
		OWL_CUDA_CHECK(cudaFreeArray(resource.surface.buffer));
	}

	lpResources_.clear();
}
auto b3d::renderer::FoveatedRenderingFeature::createResources() -> void
{
	const auto renderTargets = sharedParameters_->get<RenderTargets>("renderTargets");
	assert(renderTargets);
	inputWidth_ = renderTargets->colorRt.extent.width;
	inputHeight_ = renderTargets->colorRt.extent.height;
	const auto newWidth = static_cast<size_t>(renderTargets->colorRt.extent.width / resolutionScaleRatio_);
	const auto newHeight = static_cast<size_t>(renderTargets->colorRt.extent.height / resolutionScaleRatio_);
	lpWidth_ = newWidth;
	lpHeight_ = newHeight;
	const auto channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);


	for (auto i = 0; i < renderTargets->colorRt.extent.depth; i++)
	{
		auto lpBuffer = cudaArray_t{};
		OWL_CUDA_CHECK(cudaMallocArray(&lpBuffer, &channelDesc, newWidth, newHeight, cudaArraySurfaceLoadStore));

		const auto resourceInfo = cudaResourceDesc{ cudaResourceTypeArray, { lpBuffer } };

		auto lpSurfaceObject = cudaSurfaceObject_t{};
		OWL_CUDA_CHECK(cudaCreateSurfaceObject(&lpSurfaceObject, &resourceInfo));

		auto textureInfo = cudaTextureDesc{};
		textureInfo.addressMode[0] = cudaAddressModeClamp;
		textureInfo.addressMode[1] = cudaAddressModeClamp;

		textureInfo.filterMode = cudaFilterModeLinear;
		textureInfo.readMode = cudaReadModeNormalizedFloat;

		textureInfo.normalizedCoords = 1;
		textureInfo.maxAnisotropy = 1;
		textureInfo.maxMipmapLevelClamp = 0;
		textureInfo.minMipmapLevelClamp = 0;
		textureInfo.mipmapFilterMode = cudaFilterModePoint;
		textureInfo.borderColor[0] = 1.0f;
		textureInfo.borderColor[1] = 1.0f;
		textureInfo.borderColor[2] = 1.0f;
		textureInfo.borderColor[3] = 1.0f;
		textureInfo.sRGB = 0;
		auto lpTexture = cudaTextureObject_t{};
		OWL_CUDA_CHECK(cudaCreateTextureObject(&lpTexture, &resourceInfo, &textureInfo, nullptr));

		lpResources_.push_back({ { lpBuffer, lpSurfaceObject, newWidth, newHeight }, lpTexture });
	}
}
auto b3d::renderer::FoveatedRenderingFeature::beginUpdate() -> void
{
	const auto renderTargets = sharedParameters_->get<RenderTargets>("renderTargets");
	
	if (!renderTargets)
	{
		throw std::runtime_error("Missing shared parameters for foveated rendering feature!");
	}
	const auto width = renderTargets->colorRt.extent.width;
	const auto height = renderTargets->colorRt.extent.height;
	if (inputWidth_ != width || inputHeight_ != height)
	{
		OWL_CUDA_CHECK(cudaDeviceSynchronize());
		destroyResources();
		createResources();
	}

	assert(inputWidth_ > 0 && inputHeight_ > 0);
}
