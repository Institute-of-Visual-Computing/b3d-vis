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


#include "SharedStructs.h"
#include "owl/owl_host.h"

extern "C" char NanoRenderer_ptx[];


using namespace b3d::renderer;
using namespace b3d::renderer::nano;
using namespace owl::common;


namespace
{

	struct NanoContext
	{
		OWLContext context;
		OWLRayGen rayGen;
		OWLLaunchParams lp;

		OWLBuffer surfaceBuffer;
	};

	NanoContext nanoContext{};

	auto deleteVolume(NanoVdbVolume& volume) -> void
	{
		OWL_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(volume.grid)));
	}

	auto createVolume() -> NanoVdbVolume
	{
		// owlInstanceGroupSetTransform
		auto volume = NanoVdbVolume{};
		const auto gridVolume = nanovdb::createLevelSetTorus<float, float>(100.0f, 50.0f);
		OWL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&volume.grid), gridVolume.size()));
		OWL_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(volume.grid), gridVolume.data(), gridVolume.size(),
								  cudaMemcpyHostToDevice));

		const auto gridHandle = gridVolume.grid<float>();
		const auto& map = gridHandle->map();
		const auto orientation = owl::LinearSpace3f{ map.mMatF[0], map.mMatF[1], map.mMatF[2], map.mMatF[3], map.mMatF[4],
											   map.mMatF[5], map.mMatF[6], map.mMatF[7], map.mMatF[8] };
		const auto position = vec3f{};
		volume.transform = AffineSpace3f{orientation, position};

		{
			const auto& box = gridVolume.gridMetaData()->worldBBox();
			const auto min = owl::vec3f{ static_cast<float>(box.min()[0]), static_cast<float>(box.min()[1]),
										 static_cast<float>(box.min()[2]) };
			const auto max = owl::vec3f{ static_cast<float>(box.max()[0]), static_cast<float>(box.max()[1]),
										 static_cast<float>(box.max()[2]) };

			volume.worldAabb = owl::box3f{ min, max };
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

		return volume;
	}

	void bar()
	{
		const auto context = owlContextCreate(nullptr, 1);

		nanoContext.context = context;

		const auto module = owlModuleCreate(context, NanoRenderer_ptx);

		const auto volumeGeometryVars =
			std::array{ OWLVarDecl{ "indexBox", OWL_FLOAT3, OWL_OFFSETOF(NanoVdbVolume, indexBox) },
						OWLVarDecl{ "worldAabb", OWL_FLOAT3, OWL_OFFSETOF(NanoVdbVolume, worldAabb) },
						OWLVarDecl{ "indexBox", OWL_AFFINE3F, OWL_OFFSETOF(NanoVdbVolume, transform) },
						OWLVarDecl{ "grid", OWL_BUFFER_POINTER, OWL_OFFSETOF(NanoVdbVolume, grid) } };

		const auto geometryVars = std::array
		{
			OWLVarDecl{ "volume", OWL_USER_TYPE(NanoVdbVolume), OWL_OFFSETOF(GeometryData, volume) } 
		};

		const auto geometryType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(GeometryData),
												  geometryVars.data(), geometryVars.size());

		const auto rayGenerationVars =
			std::array{ OWLVarDecl{ "frameBufferPtr", OWL_BUFPTR, OWL_OFFSETOF(RayGenerationData, frameBufferPtr) },
						OWLVarDecl{ "frameBufferSize", OWL_INT2, OWL_OFFSETOF(RayGenerationData, frameBufferSize) },
						OWLVarDecl{ "world", OWL_GROUP, OWL_OFFSETOF(RayGenerationData, world) },
						OWLVarDecl{ "camera.position", OWL_FLOAT3, OWL_OFFSETOF(RayGenerationData, camera.position) },
						OWLVarDecl{ "camera.dir00", OWL_FLOAT3, OWL_OFFSETOF(RayGenerationData, camera.dir00) },
						OWLVarDecl{ "camera.dirDu", OWL_FLOAT3, OWL_OFFSETOF(RayGenerationData, camera.dirDu) },
						OWLVarDecl{ "camera.dirDv", OWL_FLOAT3, OWL_OFFSETOF(RayGenerationData, camera.dirDv) } };

		const auto rayGen = owlRayGenCreate(context, module, "rayGeneration", sizeof(RayGenerationData),
											rayGenerationVars.data(), rayGenerationVars.size());


		const auto launchParamsVars =
			std::array{ OWLVarDecl{ "camera", OWL_INT, OWL_OFFSETOF(LaunchParams, outputSurfaceIndex) } };

		const auto lp =
			owlParamsCreate(context, sizeof(LaunchParams), launchParamsVars.data(), launchParamsVars.size());
		nanoContext.rayGen = rayGen;
		nanoContext.lp = lp;

		auto geometry = owlGeomCreate(context, geometryType);

		const auto volume =  createVolume();



		const auto geometryGroup = owlUserGeomGroupCreate(context, 1, &geometry);
		const auto world = owlInstanceGroupCreate(context, 1, &geometryGroup,
			nullptr, nullptr, OWL_MATRIX_FORMAT_OWL, OPTIX_BUILD_FLAG_ALLOW_UPDATE);
		//owlInstanceGroupSetTransform(world, 0, (const float*)&volume.transform); 

		

		owlGeomSetRaw(geometry, "volume", &volume);

		owlGeomTypeSetBoundsProg(geometryType, module, "volumeBounds");
		owlBuildPrograms(context);

		owlGeomSetPrimCount(geometry, 1);

		owlGroupBuildAccel(geometryGroup);
		owlGroupBuildAccel(world);

		owlRayGenSetGroup(rayGen, "world", world);

		owlGeomTypeSetIntersectProg(geometryType, 0, module, "nano_intersection");
		owlGeomTypeSetClosestHit(geometryType, 0, module, "nano_closesthit");
		

		const auto missProgramVars =
			std::array{ OWLVarDecl{ "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgramData, color0) },
						OWLVarDecl{ "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgramData, color1) } };

		// ----------- create object  ----------------------------
		const auto missProgram = owlMissProgCreate(context, module, "miss", sizeof(MissProgramData),
												   missProgramVars.data(), missProgramVars.size());

		// ----------- set variables  ----------------------------
		owlMissProgSet3f(missProgram, "color0", owl3f{ .8f, 0.f, 0.f });
		owlMissProgSet3f(missProgram, "color1", owl3f{ .8f, .8f, .8f });

		
		// owlBuildProgramsDebug(context);
		owlBuildPrograms(context);
		owlBuildPipeline(context);
		// owlBuildPrograms(context)
		owlBuildSBT(context);

		nanoContext.surfaceBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(cudaSurfaceObject_t), 1, nullptr);
	}
} // namespace

auto NanoRenderer::onRender(const View& view) -> void
{

	auto waitParams = cudaExternalSemaphoreWaitParams{};
	waitParams.flags = 0;
	waitParams.params.fence.value = 0;
	cudaWaitExternalSemaphoresAsync(&initializationInfo_.signalSemaphore, &waitParams, 1);

	std::array<cudaArray_t, 2> cudaArrays{};
	std::array<cudaSurfaceObject_t, 2> cudaSurfaceObjects{};

	{
		OWL_CUDA_CHECK(cudaGraphicsMapResources(1, const_cast<cudaGraphicsResource_t*>(&view.colorRt.target)));

		for (auto i = 0; i < view.colorRt.extent.depth; i++)
		{
			OWL_CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&cudaArrays[i], view.colorRt.target, i, 0));

			auto resDesc = cudaResourceDesc{};
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cudaArrays[i];

			OWL_CUDA_CHECK(cudaCreateSurfaceObject(&cudaSurfaceObjects[i], &resDesc))
		}
	}

	owlBufferUpload(nanoContext.surfaceBuffer, cudaSurfaceObjects.data(), 0, 1);
	owlRayGenSetBuffer(nanoContext.rayGen, "frameBufferPtr", nanoContext.surfaceBuffer);

	const auto fbSize =
		owl2i{ static_cast<int32_t>(view.colorRt.extent.width), static_cast<int32_t>(view.colorRt.extent.height) };
	owlRayGenSet2i(nanoContext.rayGen, "frameBufferSize", fbSize);

	owlRayGenSet3f(nanoContext.rayGen, "camera.position", reinterpret_cast<const owl3f&>(view.cameras[0].origin));

	const auto& camera = view.cameras.front();

	auto camera_d00 = normalize(camera.at - camera.origin);
	const auto aspect = view.colorRt.extent.width / static_cast<float>(view.colorRt.extent.height);
	const auto camera_ddu = camera.cosFoV * aspect * normalize(cross(camera_d00, camera.up));
	const auto camera_ddv = camera.cosFoV * normalize(cross(camera_ddu, camera_d00));
	camera_d00 -= 0.5f * camera_ddu;
	camera_d00 -= 0.5f * camera_ddv;
	owlRayGenSet3f(nanoContext.rayGen, "camera.dir00", reinterpret_cast<const owl3f&>(camera_d00));
	owlRayGenSet3f(nanoContext.rayGen, "camera.dirDu", reinterpret_cast<const owl3f&>(camera_ddu));
	owlRayGenSet3f(nanoContext.rayGen, "camera.dirDv", reinterpret_cast<const owl3f&>(camera_ddv));

	owlBuildSBT(nanoContext.context);

	owlLaunch2D(nanoContext.rayGen, view.colorRt.extent.width, view.colorRt.extent.height, nanoContext.lp);
	// owlRayGenLaunch2D(nanoContext.rayGen, view.colorRt.extent.width, view.colorRt.extent.height);

	{
		for (auto i = 0; i < view.colorRt.extent.depth; i++)
		{
			OWL_CUDA_CHECK(cudaDestroySurfaceObject(cudaSurfaceObjects[i]));
		}
		OWL_CUDA_CHECK(cudaGraphicsUnmapResources(1, const_cast<cudaGraphicsResource_t*>(&view.colorRt.target)));
	}


	auto signalParams = cudaExternalSemaphoreSignalParams{};
	signalParams.flags = 0;
	signalParams.params.fence.value = 0;
	cudaSignalExternalSemaphoresAsync(&initializationInfo_.waitSemaphore, &signalParams, 1);
}

auto NanoRenderer::onInitialize() -> void
{
	bar();
	log("[NanoRenderer] onInitialize!");
}

auto NanoRenderer::onDeinitialize() -> void
{
	log("[NanoRenderer] onDeinitialize!");
}

auto NanoRenderer::onGui() -> void
{
}
