#include "FitsNvdbRenderer.h"

#include "owl/owl_host.h"

#include "SharedStructs.h"
#include "../../tools/fits/shared/FitsCommon.h"
#include "boost/asio/execution/context.hpp"

#include "features/RenderTargetFeature.h"
#include "features/ColorMapFeature.h"
#include "features/TransferFunctionFeature.h"

extern "C" char FitsNvdbRenderer_ptx[];
extern "C" uint8_t FitsNvdbRenderer_optixir[];
extern "C" uint32_t FitsNvdbRenderer_optixir_length;


using namespace b3d::renderer;
using namespace b3d::renderer::fitsNvdb;
using namespace owl::common;

namespace
{
	auto computeStableEpsilon(const float f) -> float
	{
		return abs(f) * static_cast<float>(1. / (1 << 21));
	}

	auto computeStableEpsilon(const vec3f& v) -> float
	{
		return max(max(computeStableEpsilon(v.x), computeStableEpsilon(v.y)), computeStableEpsilon(v.z));
	}

	auto createRayCameraData(const Camera& camera, const Extent& textureExtent) -> RayCameraData
	{
		const auto origin = vec3f{ camera.origin.x, camera.origin.y, camera.origin.z };
		const auto aspect = textureExtent.width / static_cast<float>(textureExtent.height);

		const auto vz = -normalize(camera.at - origin);
		const auto vx = normalize(cross(camera.up, vz));
		const auto vy = normalize(cross(vz, vx));
		const auto focalDistance = length(camera.at - origin);
		const auto minFocalDistance = max(computeStableEpsilon(origin), computeStableEpsilon(vx));

		const auto screenHeight = 2.f * tanf(camera.FoV / 2.f) * max(minFocalDistance, focalDistance);
		const auto vertical = screenHeight * vy;
		const auto horizontal = screenHeight * aspect * vx;
		const auto lowerLeft = -max(minFocalDistance, focalDistance) * vz - 0.5f * vertical - 0.5f * horizontal;

		return { origin, lowerLeft, horizontal, vertical };
	}
}

b3d::renderer::FitsNvdbRenderer::FitsNvdbRenderer()
{
	renderTargetFeature_ = addFeature<RenderTargetFeature>("RenderTargets");
	colorMapFeature_ = addFeature<ColorMapFeature>("Color Filtering");
	transferFunctionFeature_ = addFeature<TransferFunctionFeature>("Transfer Function");
}

auto b3d::renderer::FitsNvdbRenderer::onInitialize() -> void
{
	RendererBase::onInitialize();
	context_.owlContext = owlContextCreate(nullptr, 1);
	const auto owlContext = context_.owlContext;

	const auto irModule = owlModuleCreateFromIR(owlContext, FitsNvdbRenderer_optixir, FitsNvdbRenderer_optixir_length);
	const auto module = owlModuleCreate(owlContext, FitsNvdbRenderer_ptx);

	// Ray Generation
	// Define Variables required in Ray Generation
	// Create Ray Generation Program
	{
		const auto rayGenerationVars = std::array{
			OWLVarDecl{ "frameBufferSize", OWL_INT2, OWL_OFFSETOF(RayGenerationData, frameBufferSize) },
			OWLVarDecl{ "world", OWL_GROUP, OWL_OFFSETOF(RayGenerationData, world) },
		};
		const auto rayGen = owlRayGenCreate(owlContext, irModule, "raygen", sizeof(RayGenerationData),
											rayGenerationVars.data(), rayGenerationVars.size());
		context_.rayGen = rayGen;
	}

	// Geometry & Scene
	// Define custom geometry type and set bounds program for that type
	{
		[[maybe_unused]] const auto volumeGeometryVars =
			std::array{
			OWLVarDecl{ "fitsBox", OWL_USER_TYPE(owl::box3f), OWL_OFFSETOF(FitsNvdbGeometry, fitsBox) },
			OWLVarDecl{ "nvdbBox", OWL_USER_TYPE(owl::box3f), OWL_OFFSETOF(FitsNvdbGeometry, nvdbBox) },
		};

		const auto geometryType = owlGeomTypeCreate(owlContext, OWL_GEOM_USER, sizeof(FitsNvdbGeometry),
													volumeGeometryVars.data(), volumeGeometryVars.size());

		owlGeomTypeSetIntersectProg(geometryType, 0, irModule, "intersect");
		owlGeomTypeSetClosestHit(geometryType, 0, irModule, "closestHit");
		owlGeomTypeSetBoundsProg(geometryType, module, "bounds");
		
		// Create geometry
		owlBuildPrograms(owlContext);
		context_.geometry = owlGeomCreate(owlContext, geometryType);
		context_.geometryGroup = owlUserGeomGroupCreate(owlContext, 1, &context_.geometry);
		context_.worldGeometryGroup = owlInstanceGroupCreate(owlContext, 1, &context_.geometryGroup, nullptr, nullptr,
															 OWL_MATRIX_FORMAT_OWL, OPTIX_BUILD_FLAG_ALLOW_UPDATE);

		owlGeomSetPrimCount(context_.geometry, 1);
		owlGroupBuildAccel(context_.geometryGroup);
	}

	// Miss program
	// No variables used here
	{
		context_.missProgram = owlMissProgCreate(owlContext, irModule, "miss", 0, nullptr, 0);
	}

	// Launch parameters
	{
		const auto launchParamsVars = std::array{
			OWLVarDecl{ "cameraData", OWL_USER_TYPE(RayCameraData), OWL_OFFSETOF(LaunchParams, cameraData) },
			OWLVarDecl{ "surfacePointer", OWL_USER_TYPE(cudaSurfaceObject_t),
						OWL_OFFSETOF(LaunchParams, surfacePointer) },
			OWLVarDecl{ "bg.color0", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, bg.color0) },
			OWLVarDecl{ "bg.color1", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, bg.color1) },
			OWLVarDecl{ "bg.fillBox", OWL_BOOL, OWL_OFFSETOF(LaunchParams, bg.fillBox) },
			OWLVarDecl{ "bg.fillColor", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, bg.fillColor) },
			OWLVarDecl{ "colormaps", OWL_USER_TYPE(cudaTextureObject_t), OWL_OFFSETOF(LaunchParams, colorMaps) },
			OWLVarDecl{ "coloringInfo.colorMode", OWL_USER_TYPE(ColoringMode),
						OWL_OFFSETOF(LaunchParams, coloringInfo.coloringMode) },
			OWLVarDecl{ "coloringInfo.singleColor", OWL_FLOAT4, OWL_OFFSETOF(LaunchParams, coloringInfo.singleColor) },
			OWLVarDecl{ "coloringInfo.selectedColorMap", OWL_FLOAT,
						OWL_OFFSETOF(LaunchParams, coloringInfo.selectedColorMap) },
			OWLVarDecl{ "transferFunctionTexture", OWL_USER_TYPE(cudaTextureObject_t),
						OWL_OFFSETOF(LaunchParams, transferFunctionTexture) },
			OWLVarDecl{ "sampleRemapping", OWL_FLOAT2, OWL_OFFSETOF(LaunchParams, sampleRemapping) },
			OWLVarDecl{ "sampleIntegrationMethod", OWL_USER_TYPE(SampleIntegrationMethod),
						OWL_OFFSETOF(LaunchParams, sampleIntegrationMethod) },
			OWLVarDecl{ "volume", OWL_USER_TYPE(FitsNanoVdbVolume), OWL_OFFSETOF(LaunchParams, volume) }
		};

		context_.launchParams =
			owlParamsCreate(owlContext, sizeof(LaunchParams), launchParamsVars.data(), launchParamsVars.size());
	}

	owlBuildPrograms(owlContext);
	owlBuildPipeline(owlContext);
	

	owlGeomSetRaw(context_.geometry, "fitsBox", &fitsBox, 0);
	owlGeomSetRaw(context_.geometry, "nvdbBox", &fitsBox, 0);
	owlRayGenSetGroup(context_.rayGen, "world", context_.worldGeometryGroup);
	owlGroupBuildAccel(context_.geometryGroup);
	owlGroupBuildAccel(context_.worldGeometryGroup);

	owlBuildSBT(owlContext);
}

auto b3d::renderer::FitsNvdbRenderer::onRender() -> void
{
	// Becauase OWLBuildSBTFlags does not have NONE, we cant use the enum...
	bool buildSBT = false;

	const auto runtimeVolumeData = renderData_->get<RuntimeVolumeData>("runtimeVolumeData");

	// New Volume Available
	{
		if (runtimeVolumeData->newVolumeAvailable)
		{
			if (fitsBox != runtimeVolumeData->originalIndexBox)
			{
				fitsBox = runtimeVolumeData->originalIndexBox;
				volumeTranslateVec = -fitsBox.center();

				owlGeomSetRaw(context_.geometry, "fitsBox", &fitsBox, 0);
				owlGeomSetRaw(context_.geometry, "nvdbBox", &runtimeVolumeData->volume.volume.indexBox, 0);

				owlGroupBuildAccel(context_.geometryGroup);
				owlGroupBuildAccel(context_.worldGeometryGroup);
			}
			if (fitsBox != owl::box3f{ { 0, 0, 0 }, { 0, 0, 0 } })
			{
				const auto dims = fitsBox.size();
				runtimeVolumeData->volume.renormalizeScale =
					AffineSpace3f::scale(1.0f / max(max(dims.x, dims.y), dims.z));
			}

			const auto vol = FitsNanoVdbVolume{ runtimeVolumeData->volume.volume.grid };
			owlParamsSetRaw(context_.launchParams, "volume", &vol);
			runtimeVolumeData->newVolumeAvailable = false;
			hasVolume_ = true;
		}
	}
	if (!hasVolume_)
	{
		return;
	}

	const auto volumeTransform = renderData_->get<VolumeTransform>("volumeTransform");

	// Refit and transform the volume
	{
		const auto volumeTranslate = owl::AffineSpace3f::translate(volumeTranslateVec);

		trs_ = volumeTransform->worldMatTRS * runtimeVolumeData->volume.renormalizeScale;
		const auto groupTransform = trs_ * volumeTranslate;

		owlInstanceGroupSetTransform(context_.worldGeometryGroup, 0, reinterpret_cast<const float*>(&groupTransform));
		owlGroupRefitAccel(context_.worldGeometryGroup);
	
		debugDraw().drawBox(trs_.p / 2, 0, fitsBox.size(), owl::vec4f(0.1f, 0.82f, 0.15f, 1.0f), trs_.l);
		debugInfo_.gizmoHelper->drawGizmo(volumeTransform->worldMatTRS);
	}

	auto renderTargetFeatureParams = renderTargetFeature_->getParamsData();
	const auto framebufferSize = owl::vec2i{ static_cast<int32_t>(renderTargetFeatureParams.colorRT.extent.width),
										static_cast<int32_t>(renderTargetFeatureParams.colorRT.extent.height) };

	const auto view = renderData_->get<View>("view");

	assert(view->cameras.size() > 0);
	const auto cameraIndices = view->mode == RenderMode::stereo ? std::vector{ 0, 1 } : std::vector{ 0 };

	auto rayCameraData = std::array<RayCameraData, 2>{};
	// Camera Setup
	{
		assert(view->mode == RenderMode::stereo ? renderTargetFeatureParams.colorRT.extent.depth > 1 : true);
		for (const auto cameraIndex : cameraIndices)
		{
			assert(cameraIndex < view->cameras.size());
			const auto& camera = view->cameras[cameraIndex];

			if (camera.directionsAvailable)
			{
				rayCameraData[cameraIndex] = { camera.origin, camera.dir00, camera.dirDu, camera.dirDv };
			}
			else
			{
				rayCameraData[cameraIndex] = createRayCameraData(
					camera,
					Extent{ static_cast<uint32_t>(framebufferSize.x), static_cast<uint32_t>(framebufferSize.y),
							1 } /*renderTargetFeatureParams.colorRT.extent*/);
			}
		}
	}


	if (currentFramebufferSize != framebufferSize)
	{
		currentFramebufferSize = framebufferSize;
		owlRayGenSet2i(context_.rayGen, "frameBufferSize", currentFramebufferSize.x, currentFramebufferSize.y);
		buildSBT = true;
	}

	// Transferfunction
	{
		auto transferFunctionParams = transferFunctionFeature_->getParamsData();
		owlParamsSetRaw(context_.launchParams, "transferFunctionTexture",
						&transferFunctionParams.transferFunctionTexture);
	}
	// Colormap
	{
		const auto coloringInfo = renderData_->get<ColoringInfo>("coloringInfo");
		owlParamsSetRaw(context_.launchParams, "coloringInfo.colorMode", &coloringInfo->coloringMode);
		owlParamsSet4f(context_.launchParams, "coloringInfo.singleColor", coloringInfo->singleColor.x,
					   coloringInfo->singleColor.y, coloringInfo->singleColor.z, coloringInfo->singleColor.w);

		owlParamsSet1f(context_.launchParams, "coloringInfo.selectedColorMap", coloringInfo->selectedColorMap);

		const auto colorMapParams = colorMapFeature_->getParamsData();
		owlParamsSetRaw(context_.launchParams, "colormaps", &colorMapParams.colorMapTexture);
	}



	// std::array<float, 2> sampleRemapping{ 0.0f, 0.1f }
	owlParamsSet2f(context_.launchParams, "sampleRemapping", owl2f{ 0.0f, 0.1f });
	const auto sampleIntegrationMethod = SampleIntegrationMethod::maximumIntensityProjection;
	owlParamsSetRaw(context_.launchParams, "sampleIntegrationMethod", &sampleIntegrationMethod);

	owlParamsSetRaw(context_.launchParams, "cameraData", &rayCameraData[0]);
	owlParamsSetRaw(context_.launchParams, "surfacePointer", &renderTargetFeatureParams.colorRT.surfaces[0].surface);


	if (buildSBT)
	{
		owlBuildSBT(context_.owlContext);	
	}

	owlAsyncLaunch2D(context_.rayGen, framebufferSize.x, framebufferSize.y, context_.launchParams);
}

auto b3d::renderer::FitsNvdbRenderer::onDeinitialize() -> void
{
	owlContextDestroy(context_.owlContext);
}
