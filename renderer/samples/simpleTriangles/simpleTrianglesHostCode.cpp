#include "RendererBase.h"

#include "SimpleTrianglesRenderer.h"

// public owl node-graph API
#include "owl/owl.h"
// our device-side data structures
#include <cuda/std/cstddef>
#include <filesystem>


#include "imgui.h"
#include "owl/helper/cuda.h"


#include "stb_image.h"

#include "ColorMap.h"

#include "deviceCode.h"

using namespace b3d::renderer;

extern "C" char SimpleTrianglesDeviceCode_ptx[];

namespace
{
	float computeStableEpsilon(float f)
	{
		return abs(f) * float(1. / (1 << 21));
	}

	float computeStableEpsilon(const vec3f v)
	{
		return max(max(computeStableEpsilon(v.x), computeStableEpsilon(v.y)), computeStableEpsilon(v.z));
	}


	std::string coloringModeStrings[2] = { "Single", "ColorMap" };

	struct GuiData
	{
		struct BackgroundColorPalette
		{
			std::array<float, 4> color1{ 0.572f, 0.100f, 0.750f, 1.0f };
			std::array<float, 4> color2{ 0.0f, 0.3f, 0.3f, 1.0f };
		};

		BackgroundColorPalette rtBackgroundColorPalette;
		std::array<float, 4> singleColor{ 0, 1, 0, 1 };

		int coloringModeInt = 0;
		int selectedColorMap = 0;
	};

	RayCameraData createRayCameraData(const Camera& camera, const Extent& textureExtent)
	{
		const auto origin = vec3f{ camera.origin.x, camera.origin.y, camera.origin.z };
		auto camera_d00 = camera.at - origin;
		const auto wlen = length(camera_d00);

		const auto aspect = textureExtent.width / static_cast<float>(textureExtent.height);
		const auto vlen = wlen * std::cosf(0.5f * camera.FoV);
		const auto camera_ddu = vlen * aspect * normalize(cross(camera_d00, camera.up));
		const auto camera_ddv = vlen * normalize(cross(camera_ddu, camera_d00));

		const auto vz = -normalize(camera.at - origin);
		const auto vx = normalize(cross(camera.up, vz));
		const auto vy = normalize(cross(vz, vx));
		const auto focalDistance = length(camera.at - origin);
		const float minFocalDistance = max(computeStableEpsilon(origin), computeStableEpsilon(vx));

		float screen_height = 2.f * tanf(camera.FoV / 2.f) * max(minFocalDistance, focalDistance);
		const auto vertical = screen_height * vy;
		const auto horizontal = screen_height * aspect * vx;
		const auto lower_left = -max(minFocalDistance, focalDistance) * vz - 0.5f * vertical - 0.5f * horizontal;

		return { origin, lower_left, horizontal, vertical };
	}

	GuiData guiData{};


	const int NUM_VERTICES = 8;
	vec3f vertices[NUM_VERTICES] = { { -0.5f, -0.5f, -0.5f }, { +0.5f, -0.5f, -0.5f }, { -0.5f, +0.5f, -0.5f },
									 { +0.5f, +0.5f, -0.5f }, { -0.5f, -0.5f, +0.5f }, { +0.5f, -0.5f, +0.5f },
									 { -0.5f, +0.5f, +0.5f }, { +0.5f, +0.5f, +0.5f } };

	const int NUM_INDICES = 12;
	vec3i indices[NUM_INDICES] = { { 0, 1, 3 }, { 2, 3, 0 }, { 5, 7, 6 }, { 5, 6, 4 }, { 0, 4, 5 }, { 0, 5, 1 },
								   { 2, 3, 7 }, { 2, 7, 6 }, { 1, 5, 7 }, { 1, 7, 3 }, { 4, 0, 2 }, { 4, 2, 6 } };

	// TODO: Mapping wrong
	vec2f texCoords[NUM_VERTICES] = {

		{ +0.f, +0.f }, { +0.f, +1.f }, { +1.f, +0.f }, { +1.f, +1.f },

		{ +0.f, +0.f }, { +0.f, +1.f }, { +1.f, +0.f }, { +1.f, +1.f },
	};

} // namespace

auto SimpleTrianglesRenderer::onRender() -> void
{

	const auto synchronization = renderData_->get<Synchronization>("synchronization");

	auto waitParams = cudaExternalSemaphoreWaitParams{};
	waitParams.flags = 0;
	waitParams.params.fence.value = synchronization->fenceValue;
	cudaWaitExternalSemaphoresAsync(&synchronization->signalSemaphore, &waitParams, 1);

	// map/create/set surface
	std::array<cudaArray_t, 2> cudaArrays{};
	std::array<cudaSurfaceObject_t, 2> cudaSurfaceObjects{};

	
	const auto renderTargets = renderData_->get<RenderTargets>("renderTargets");

	auto cudaRet = cudaSuccess;
	// Map and createSurface
	{
		cudaRet = cudaGraphicsMapResources(1, &renderTargets->colorRt.target);
		for (auto i = 0; i < renderTargets->colorRt.extent.depth; i++)
		{
			cudaRet = cudaGraphicsSubResourceGetMappedArray(&cudaArrays[i], renderTargets->colorRt.target, i, 0);

			cudaResourceDesc resDesc{};
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cudaArrays[i];
			cudaRet = cudaCreateSurfaceObject(&cudaSurfaceObjects[i], &resDesc);
		}
	}

	const auto colorMapTexture = renderData_->get<ExternalTexture>("colorMapTexture");
	const auto coloringInfo = renderData_->get<ColoringInfo>("coloringInfo");
	cudaArray_t colorMapCudaArray{};
	cudaTextureObject_t colorMapCudaTexture{};
	{
		OWL_CUDA_CHECK(cudaGraphicsMapResources(1, &colorMapTexture->target));

		OWL_CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&colorMapCudaArray, colorMapTexture->target, 0, 0));

		// Create texture
		auto resDesc = cudaResourceDesc{};
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = colorMapCudaArray;

		auto texDesc = cudaTextureDesc{};
		texDesc.addressMode[0] = cudaAddressModeClamp;
		texDesc.addressMode[1] = cudaAddressModeClamp;

		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType; // cudaReadModeNormalizedFloat

		texDesc.normalizedCoords = 1;
		texDesc.maxAnisotropy = 1;
		texDesc.maxMipmapLevelClamp = 0;
		texDesc.minMipmapLevelClamp = 0;
		texDesc.mipmapFilterMode = cudaFilterModePoint;
		texDesc.borderColor[0] = 1.0f;
		texDesc.borderColor[1] = 1.0f;
		texDesc.borderColor[2] = 1.0f;
		texDesc.borderColor[3] = 1.0f;
		texDesc.sRGB = 0;

		OWL_CUDA_CHECK(cudaCreateTextureObject(&colorMapCudaTexture, &resDesc, &texDesc, nullptr));
	}

	if (fbSize_.x != renderTargets->colorRt.extent.width || fbSize_.y != renderTargets->colorRt.extent.height)
	{
		fbSize_ = { static_cast<int32_t>(renderTargets->colorRt.extent.width),
					static_cast<int32_t>(renderTargets->colorRt.extent.height) };
		owlRayGenSet2i(rayGen_, "fbSize", fbSize_);
		sbtDirty = true;
	}

	if (sbtDirty)
	{
		owlBuildSBT(context_);
		sbtDirty = false;
	}

	// Instance transform
	{
		const auto volumeTransform = renderData_->get<VolumeTransform>("volumeTransform");
		owlInstanceGroupSetTransform(world_, 0, (const float*)&volumeTransform->worldMatTRS);
		owlGroupRefitAccel(world_);
	}

	// const auto coloringInfo = renderData_->get<ColoringInfo>("coloringInfo");

	owlParamsSetRaw(launchParameters_, "coloringInfo.colorMode", &coloringInfo->coloringMode);
	owlParamsSet4f(
		launchParameters_, "coloringInfo.singleColor",
				   { coloringInfo->singleColor.x, coloringInfo->singleColor.y, coloringInfo->singleColor.z,
					 coloringInfo->singleColor.w });
	owlParamsSet1f(launchParameters_, "coloringInfo.selectedColorMap", coloringInfo->selectedColorMap);


	// Regular use would be to use owlParamsSet4f but im to lazy to cast color1 to owl4f

	owlParamsSet4f(launchParameters_, "backgroundColor0",
				   { guiData.rtBackgroundColorPalette.color1[0], guiData.rtBackgroundColorPalette.color1[1],
					 guiData.rtBackgroundColorPalette.color1[2], guiData.rtBackgroundColorPalette.color1[3] });
	owlParamsSet4f(launchParameters_, "backgroundColor1",
				   { guiData.rtBackgroundColorPalette.color2[0], guiData.rtBackgroundColorPalette.color2[1],
					 guiData.rtBackgroundColorPalette.color2[2], guiData.rtBackgroundColorPalette.color2[3] });
	owlParamsSetRaw(launchParameters_, "colormaps", &colorMapCudaTexture);

	const auto view = renderData_->get<View>("view");
	{
		RayCameraData rcd;
		if (view->cameras[0].directionsAvailable)
		{
			rcd = { view->cameras[0].origin, view->cameras[0].dir00, view->cameras[0].dirDu, view->cameras[0].dirDv };
		}
		else
		{
			rcd = createRayCameraData(view->cameras[0], renderTargets->colorRt.extent);
		}

		owlParamsSetRaw(launchParameters_, "cameraData", &rcd);
		owlParamsSetRaw(launchParameters_, "surfacePointer", &cudaSurfaceObjects[0]);

		owlAsyncLaunch2D(rayGen_, renderTargets->colorRt.extent.width, renderTargets->colorRt.extent.height,
						 launchParameters_);
	}

	if (view->mode == RenderMode::stereo && renderTargets->colorRt.extent.depth > 1)
	{
		RayCameraData rcd;
		if (view->cameras[1].directionsAvailable)
		{
			rcd = { view->cameras[1].origin, view->cameras[1].dir00, view->cameras[1].dirDu, view->cameras[1].dirDv };
		}
		else
		{
			rcd = createRayCameraData(view->cameras[1], renderTargets->colorRt.extent);
		}
		owlParamsSetRaw(launchParameters_, "cameraData", &rcd);
		owlParamsSetRaw(launchParameters_, "surfacePointer", &cudaSurfaceObjects[1]);
		owlAsyncLaunch2D(rayGen_, renderTargets->colorRt.extent.width, renderTargets->colorRt.extent.height,
						 launchParameters_);
	}


	{
		for (auto i = 0; i < renderTargets->colorRt.extent.depth; i++)
		{
			cudaRet = cudaDestroySurfaceObject(cudaSurfaceObjects[i]);
		}
		cudaRet = cudaGraphicsUnmapResources(1, const_cast<cudaGraphicsResource_t*>(&renderTargets->colorRt.target));
	}

	{
		OWL_CUDA_CHECK(cudaDestroyTextureObject(colorMapCudaTexture));
		cudaGraphicsUnmapResources(1, &colorMapTexture->target);
	}

	auto signalParams = cudaExternalSemaphoreSignalParams{};
	signalParams.flags = 0;
	signalParams.params.fence.value = synchronization->fenceValue;
	cudaSignalExternalSemaphoresAsync(&synchronization->waitSemaphore, &signalParams, 1);
}

auto SimpleTrianglesRenderer::onInitialize() -> void
{
	RendererBase::onInitialize();

	// create a context on the first device:
	context_ = owlContextCreate(nullptr, 1);
	auto module = owlModuleCreate(context_, SimpleTrianglesDeviceCode_ptx);


	// ##################################################################
	// set up all the *GEOMETRY* graph we want to render
	// ##################################################################

	// -------------------------------------------------------
	// declare geometry type
	// -------------------------------------------------------
	OWLVarDecl trianglesGeomVars[] = {
		{ "color", OWL_FLOAT4, OWL_OFFSETOF(TrianglesGeomData, color) },
		{ "index", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index) },
		{ "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex) },
		{ "texCoord", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, texCoord) },
		{}
	};
	OWLGeomType trianglesGeomType =
		owlGeomTypeCreate(context_, OWL_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars, -1);
	owlGeomTypeSetClosestHit(trianglesGeomType, 0, module, "TriangleMesh");

	// ##################################################################
	// set up all the *GEOMS* we want to run that code on
	// ##################################################################

	// ------------------------------------------------------------------
	// triangle mesh
	// ------------------------------------------------------------------
	OWLBuffer vertexBuffer = owlDeviceBufferCreate(context_, OWL_FLOAT3, NUM_VERTICES, vertices);
	OWLBuffer indexBuffer = owlDeviceBufferCreate(context_, OWL_INT3, NUM_INDICES, indices);

	OWLBuffer texCoordsBuffer = owlDeviceBufferCreate(context_, OWL_FLOAT2, NUM_VERTICES, texCoords);

	OWLGeom trianglesGeom = owlGeomCreate(context_, trianglesGeomType);

	owlTrianglesSetVertices(trianglesGeom, vertexBuffer, NUM_VERTICES, sizeof(vec3f), 0);
	owlTrianglesSetIndices(trianglesGeom, indexBuffer, NUM_INDICES, sizeof(vec3i), 0);

	owlGeomSetBuffer(trianglesGeom, "vertex", vertexBuffer);
	owlGeomSetBuffer(trianglesGeom, "index", indexBuffer);

	owlGeomSetBuffer(trianglesGeom, "texCoord", texCoordsBuffer);
	owlGeomSet4f(trianglesGeom, "color", owl4f{ 0, 1, 0, 1 });


	// ------------------------------------------------------------------
	// the group/accel for that mesh
	// ------------------------------------------------------------------
	OWLGroup trianglesGroup = owlTrianglesGeomGroupCreate(context_, 1, &trianglesGeom);
	owlGroupBuildAccel(trianglesGroup);
	world_ = owlInstanceGroupCreate(context_, 1, &trianglesGroup, nullptr, nullptr, OWL_MATRIX_FORMAT_OWL,
									OPTIX_BUILD_FLAG_ALLOW_UPDATE);
	owlGroupBuildAccel(world_);

	// -------------------------------------------------------
	// set up miss prog
	// -------------------------------------------------------
	{
		OWLVarDecl missProgVars[] = { { "color0", OWL_FLOAT4, OWL_OFFSETOF(MissProgData, color0) },
									  { "color1", OWL_FLOAT4, OWL_OFFSETOF(MissProgData, color1) },
									  { /* sentinel to mark end of list */ } };
		// ----------- create object  ----------------------------
		missProg_ = owlMissProgCreate(context_, module, "miss", sizeof(MissProgData), missProgVars, -1);

		// ----------- set variables  ----------------------------
		owlMissProgSet4f(missProg_, "color0", owl4f{ .0f, 0.f, 0.f, 0.0f });
		owlMissProgSet4f(missProg_, "color1", owl4f{ .0f, .0f, .0f, 0.0f });
	}

	// -------------------------------------------------------
	// set up ray gen program
	// -------------------------------------------------------
	{
		OWLVarDecl rayGenVars[] = { { "fbSize", OWL_INT2, OWL_OFFSETOF(RayGenData, fbSize) },
									{ "world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world) },
									{ /* sentinel to mark end of list */ } };

		rayGen_ = owlRayGenCreate(context_, module, "simpleRayGen", sizeof(RayGenData), rayGenVars, -1);

		owlRayGenSetGroup(rayGen_, "world", world_);
	}

	// -------------------------------------------------------
	// set up launch params
	// -------------------------------------------------------
	{
		OWLVarDecl launchParamsVarsWithStruct[] = {
			{ "cameraData", OWL_USER_TYPE(RayCameraData), OWL_OFFSETOF(MyLaunchParams, cameraData) },
			{ "surfacePointer", OWL_USER_TYPE(cudaSurfaceObject_t), OWL_OFFSETOF(MyLaunchParams, surfacePointer) },
			{ "colormaps", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(MyLaunchParams, colorMaps) },
			{ "coloringInfo.colorMode", OWL_USER_TYPE(ColoringMode), OWL_OFFSETOF(MyLaunchParams, coloringInfo.coloringMode) },
			{ "coloringInfo.singleColor", OWL_FLOAT4, OWL_OFFSETOF(MyLaunchParams, coloringInfo.singleColor) },
			{ "coloringInfo.selectedColorMap", OWL_FLOAT, OWL_OFFSETOF(MyLaunchParams, coloringInfo.selectedColorMap) },
			{ "backgroundColor0", OWL_FLOAT4, OWL_OFFSETOF(MyLaunchParams, backgroundColor0) },
			{ "backgroundColor1", OWL_FLOAT4, OWL_OFFSETOF(MyLaunchParams, backgroundColor1) },
			{}
		};

		launchParameters_ = owlParamsCreate(context_, sizeof(MyLaunchParams), launchParamsVarsWithStruct, -1);
	}

	owlBuildPrograms(context_);
	owlBuildPipeline(context_);
	owlBuildSBT(context_);
}

auto SimpleTrianglesRenderer::onGui() -> void
{
	const auto volumeTransform = renderData_->get<VolumeTransform>("volumeTransform");
	const auto coloringInfo = renderData_->get<ColoringInfo>("coloringInfo");
	const auto colorMapInfos = renderData_->get<ColorMapInfos>("colorMapInfos");

	ImGui::Begin("RT Settings");

	ImGui::SeparatorText("Coloring");

	ImGui::Combo("Mode", &guiData.coloringModeInt, "Single\0ColorMap\0\0");
	if (guiData.coloringModeInt == 0)
	{
		ImGui::ColorEdit3("Color", &coloringInfo->singleColor.x);
	}
	else
	{
		if (ImGui::BeginCombo("combo 1", (*colorMapInfos->colorMapNames)[guiData.selectedColorMap].c_str(), 0))
		{
			for (int n = 0; n < colorMapInfos->colorMapNames->size(); n++)
			{
				const bool is_selected = (guiData.selectedColorMap == n);
				if (ImGui::Selectable((*colorMapInfos->colorMapNames)[n].c_str(), is_selected))
				{
					guiData.selectedColorMap = n;
				}

				// Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
				if (is_selected)
					ImGui::SetItemDefaultFocus();
			}
			ImGui::EndCombo();
		}
	}

	if (ImGui::Button("Select"))
	{
		ImGui::OpenPopup("FileSelectDialog");
	}

	ImGui::SeparatorText("Background Color Palette");
	ImGui::ColorEdit3("Color 1", guiData.rtBackgroundColorPalette.color1.data());
	ImGui::ColorEdit3("Color 2", guiData.rtBackgroundColorPalette.color2.data());
	debugInfo_.gizmoHelper->drawGizmo(volumeTransform->worldMatTRS);

	ImGui::End();

	coloringInfo->coloringMode = guiData.coloringModeInt == 0 ? single : colormap;
	coloringInfo->selectedColorMap = colorMapInfos->firstColorMapYTextureCoordinate +
		static_cast<float>(guiData.selectedColorMap) * colorMapInfos->colorMapHeightNormalized;
}
