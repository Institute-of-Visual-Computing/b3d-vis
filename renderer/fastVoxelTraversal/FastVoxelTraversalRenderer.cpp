
#include "FastVoxelTraversalRenderer.h"


#include <filesystem>

#include "FastVoxelTraversalSharedStructs.h"

#include "owl/owl.h"

#include "imgui.h"
#include "owl/helper/cuda.h"

#include "Logging.h"
#include "SourceVolumeLoader.h"

#include "stb_image.h"

using namespace b3d::renderer;

extern "C" char FastVoxelTraversalDeviceCode_ptx[];
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
} // namespace

void FastVoxelTraversalRenderer::onInitialize()
{
	RendererBase::onInitialize();

	const auto fitsFilePathS = std::filesystem::path{ "D:/data/work/b3d_data/datacubes/n4565/n4565_lincube_big.fits" };
	const auto catalogFilePathS =
		std::filesystem::path{ "D:/data/work/b3d_data/datacubes/n4565/sofia_output/outname_cat.xml" };
	const auto transferFunction1DFilePath = std::filesystem::path{ "resources/transfer1d.png" };

	context_ = owlContextCreate(nullptr, 1);
	auto module = owlModuleCreate(context_, FastVoxelTraversalDeviceCode_ptx);

	// Raygen
	{
		OWLVarDecl rayGenVars[] = { { "fbSize", OWL_INT2, OWL_OFFSETOF(RayGenData, fbSize) },
									{ "world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world) },
									{ /* sentinel to mark end of list */ } };

		rayGen_ = owlRayGenCreate(context_, module, "simpleRayGen", sizeof(RayGenData), rayGenVars, -1);
	}

	// Miss Program
	{
		OWLVarDecl missProgVars[] = { { "color0", OWL_FLOAT4, OWL_OFFSETOF(MissProgData, color0) },
									  { "color1", OWL_FLOAT4, OWL_OFFSETOF(MissProgData, color1) },
									  {} };

		missProg_ = owlMissProgCreate(context_, module, "miss", sizeof(MissProgData), missProgVars, -1);
	}

	// Launch parameters
	{
		OWLVarDecl launchParamsVarsWithStruct[] = {
			{ "cameraData", OWL_USER_TYPE(RayCameraData), OWL_OFFSETOF(MyLaunchParams, cameraData) },
			{ "surfacePointer", OWL_USER_TYPE(cudaSurfaceObject_t), OWL_OFFSETOF(MyLaunchParams, surfacePointer) },
			{ "transferTexture1D", OWL_TEXTURE_2D, OWL_OFFSETOF(MyLaunchParams, transferTexture1D) },
			{ "transferOffset", OWL_FLOAT, OWL_OFFSETOF(MyLaunchParams, transferOffset) },
			{ "integralValue", OWL_FLOAT, OWL_OFFSETOF(MyLaunchParams, integralValue) },
			{ "inverseIntegralValue", OWL_FLOAT, OWL_OFFSETOF(MyLaunchParams, inverseIntegralValue) },
			{}
		};

		launchParameters_ = owlParamsCreate(context_, sizeof(MyLaunchParams), launchParamsVarsWithStruct, -1);
	}

	// Create Geom Type
	{
		OWLVarDecl aabbGeomsVar[] = { { "sourceRegions", OWL_BUFPTR, OWL_OFFSETOF(DatacubeSources, sourceRegions) },
									  { "gridData", OWL_BUFPTR, OWL_OFFSETOF(DatacubeSources, gridData) },
									  { "gridDims", OWL_INT3, OWL_OFFSETOF(DatacubeSources, gridDims) },
									  { "minmax", OWL_FLOAT2, OWL_OFFSETOF(DatacubeSources, minmax) },
									  { /* sentinel to mark end of list */ } };

		aabbGeomType_ = owlGeomTypeCreate(context_, OWL_GEOM_USER, sizeof(DatacubeSources), aabbGeomsVar, -1);

		owlGeomTypeSetBoundsProg(aabbGeomType_, module, "AABBGeom");
		owlGeomTypeSetIntersectProg(aabbGeomType_, 0, module, "AABBGeom");
		owlGeomTypeSetClosestHit(aabbGeomType_, 0, module, "AABBGeom");
		owlGeomTypeSetAnyHit(aabbGeomType_, 0, module, "AABBGeom");

		concreteAabbGeom_ = owlGeomCreate(context_, aabbGeomType_);
	}

	// Load transferfunction image
	{
		int x, y, n;

		unsigned char* data = stbi_load(transferFunction1DFilePath.string().c_str(), &x, &y, &n, 0);

		integral_ = 0;
		for (int i = 0; i < x * y; ++i)
		{
			integral_ += static_cast<float>(data[i]);
		}
		integral_ /= 255.0f;
		integral_ /= static_cast<float>(x * y);
		invIntegral_ = 1.0f / integral_;

		transferTexture1D_ =
			owlTexture2DCreate(context_, OWL_TEXEL_FORMAT_RGBA8, x, y, data, OWL_TEXTURE_LINEAR, OWL_TEXTURE_WRAP);
	}

	// Load Sources and create geomgroup from aabbgeomtype
	{
		std::vector<SourceRegion> sourceRegions;
		std::vector<float> sourcesDataBuffer;

		const auto bufferSize =
			SourceVolumeLoader::extractSourceRegionsFromCatalogueXML(catalogFilePathS.string(), sourceRegions);
		sourcesDataBuffer.resize(bufferSize);
		const auto volumeDimension =
			SourceVolumeLoader::loadDataForSources(fitsFilePathS.string(), sourceRegions, sourcesDataBuffer);

		std::sort(sourceRegions.begin(), sourceRegions.end(),
				  [](const SourceRegion& a, const SourceRegion& b)
				  { return a.gridSourceBox.volume() < b.gridSourceBox.volume(); });


		const owl2f gridDataMinMax = { *std::min_element(sourcesDataBuffer.begin(), sourcesDataBuffer.end()),
									   *std::max_element(sourcesDataBuffer.begin(), sourcesDataBuffer.end()) };


		OWLBuffer owlSourcesBuffer =
			owlDeviceBufferCreate(context_, OWL_USER_TYPE(SourceRegion), sourceRegions.size(), sourceRegions.data());
		OWLBuffer owlDataBuffer =
			owlDeviceBufferCreate(context_, OWL_FLOAT, sourcesDataBuffer.size(), sourcesDataBuffer.data());


		// Set data to aabgeom
		owlGeomSetBuffer(concreteAabbGeom_, "sourceRegions", owlSourcesBuffer);
		owlGeomSetBuffer(concreteAabbGeom_, "gridData", owlDataBuffer);
		owlGeomSet3i(concreteAabbGeom_, "gridDims", owl3i{ volumeDimension.x, volumeDimension.y, volumeDimension.z });
		owlGeomSet2f(concreteAabbGeom_, "minmax", gridDataMinMax);
		owlGeomSetPrimCount(concreteAabbGeom_, sourceRegions.size());

		aabbGroup_ = owlUserGeomGroupCreate(context_, 1, &concreteAabbGeom_);
	}

	// Build everything and set variables

	owlBuildPrograms(context_);
	owlBuildPipeline(context_);
	owlGroupBuildAccel(aabbGroup_);

	world_ = owlInstanceGroupCreate(context_, 1, &aabbGroup_, nullptr, nullptr, OWL_MATRIX_FORMAT_OWL,
									OPTIX_BUILD_FLAG_ALLOW_UPDATE);

	owlParamsSetTexture(launchParameters_, "transferTexture1D", transferTexture1D_);
	owlParamsSet1f(launchParameters_, "integralValue", integral_);
	owlParamsSet1f(launchParameters_, "inverseIntegralValue", invIntegral_);

	owlMissProgSet4f(missProg_, "color0", owl4f{ .8f, 0.f, 0.f, 1.0f });
	owlMissProgSet4f(missProg_, "color1", owl4f{ .8f, .8f, .8f, 1.0f });

	owlGroupBuildAccel(world_);
	owlRayGenSetGroup(rayGen_, "world", world_);
	owlBuildSBT(context_);
}

auto FastVoxelTraversalRenderer::onRender() -> void
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

	const auto volumeTransform = renderData_->get<VolumeTransform>("volumeTransform");
	const auto transferOffset = renderData_->get<float>("transferOffset");

	const auto view = renderData_->get<View>("view");

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


	owlInstanceGroupSetTransform(world_, 0, (const float*)&volumeTransform->worldMatTRS);
	owlGroupRefitAccel(world_);

	// Set Launch Params for this run. 
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
		owlParamsSet1f(launchParameters_, "transferOffset", *transferOffset);
	}
	
	owlAsyncLaunch2D(rayGen_, renderTargets->colorRt.extent.width, renderTargets->colorRt.extent.height,
					 launchParameters_);


	// Destroy and unmap surface
	{
		for (auto i = 0; i < renderTargets->colorRt.extent.depth; i++)
		{
			cudaRet = cudaDestroySurfaceObject(cudaSurfaceObjects[i]);
		}
		cudaRet = cudaGraphicsUnmapResources(1, const_cast<cudaGraphicsResource_t*>(&renderTargets->colorRt.target));
	}

	auto signalParams = cudaExternalSemaphoreSignalParams{};
	signalParams.flags = 0;
	signalParams.params.fence.value = synchronization->fenceValue;
	cudaSignalExternalSemaphoresAsync(&synchronization->waitSemaphore, &signalParams, 1);
}

void FastVoxelTraversalRenderer::onDeinitialize()
{
	RendererBase::onDeinitialize();
}

void FastVoxelTraversalRenderer::onGui()
{

	const auto volumeTransform = renderData_->get<VolumeTransform>("volumeTransform");
	const auto transferOffset = renderData_->get<float>("transferOffset");

	ImGui::Begin("RT Settings");
	ImGui::SeparatorText("Transfer Offset");

	ImGui::DragFloat("Transfer Offset", transferOffset, 0.001f, 0.0f, 1.0f);

	debugInfo_.gizmoHelper->drawGizmo(volumeTransform->worldMatTRS);

	ImGui::End();
}
