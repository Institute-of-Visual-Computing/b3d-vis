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

	auto foo() -> void
	{
		auto volume = NanoVdbVolume{};
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
						OWLVarDecl{ "indexBox", OWL_BUFFER_POINTER, OWL_OFFSETOF(NanoVdbVolume, volume) } };

		const auto volumeType = owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(NanoVdbVolume),
												  volumeGeometryVars.data(), volumeGeometryVars.size());


		const auto rayGenerationVars =
			std::array{ OWLVarDecl{ "frameBufferPtr", OWL_BUFPTR,
									OWL_OFFSETOF(RayGenerationData, frameBufferPtr) },
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

		const auto lp = owlParamsCreate(context, sizeof(LaunchParams), launchParamsVars.data(), launchParamsVars.size());
		nanoContext.rayGen = rayGen;
		nanoContext.lp = lp;

		auto geometry = owlGeomCreate(context, volumeType);

		const auto geometryGroup = owlUserGeomGroupCreate(context, 1, &geometry);
		const auto world = owlInstanceGroupCreate(context, 1, &geometryGroup, nullptr, nullptr, OWL_MATRIX_FORMAT_OWL);


		owlGeomTypeSetBoundsProg(volumeType, module, "volumeBounds");
		owlBuildProgramsDebug(context);

		owlGeomSetPrimCount(geometry, 1);

		owlGroupBuildAccel(geometryGroup);
		owlGroupBuildAccel(world);

		owlRayGenSetGroup(rayGen, "world", world);

		// Programs for user geom must be set explicit
		
		// owlGeomTypeSetIntersectProg(aabbGeomType, 0, module, "AABBGeom");
		// owlGeomTypeSetClosestHit(aabbGeomType, 0, module, "AABBGeom");
		//  owlGeomTypeSetAnyHit(aabbGeomType, 0, module, "AABBGeom");
		//owlBuildPrograms(context);
		//owlBuildProgramsDebug(context);
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
		{
			const auto error = cudaGraphicsMapResources(1, const_cast<cudaGraphicsResource_t*>(&view.colorRt.target));
		}
		for (auto i = 0; i < view.colorRt.extent.depth; i++)
		{
			{
				const auto error = cudaGraphicsSubResourceGetMappedArray(&cudaArrays[i], view.colorRt.target, i, 0);
			}

			auto resDesc = cudaResourceDesc{};
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = cudaArrays[i];
			{
				const auto error = cudaCreateSurfaceObject(&cudaSurfaceObjects[i], &resDesc);
			}
		}
	}


	owlBufferUpload(nanoContext.surfaceBuffer, cudaSurfaceObjects.data(), 0, 1);

	owlRayGenSetBuffer(nanoContext.rayGen, "frameBufferPtr", nanoContext.surfaceBuffer);

	const auto fbSize = owl2i{
		static_cast<int32_t>(view.colorRt.extent.width),
		static_cast<int32_t>(view.colorRt.extent.height)
	};
	owlRayGenSet2i(nanoContext.rayGen, "frameBufferSize", fbSize);
	
	owlRayGenSet3f(nanoContext.rayGen, "camera.position", reinterpret_cast<const owl3f&>(view.cameras[0].origin));
	

	const vec3f lookFrom = view.cameras[0].origin;
	const vec3f lookAt = view.cameras[0].at;
	const vec3f lookUp = view.cameras[0].up;
	const float cosFovy = view.cameras[0].cosFoV;

	// ----------- compute variable values  ------------------
	vec3f camera_pos = lookFrom;
	vec3f camera_d00 = normalize(lookAt - lookFrom);
	float aspect = view.colorRt.extent.width / static_cast<float>(view.colorRt.extent.height);
	vec3f camera_ddu = cosFovy * aspect * normalize(cross(camera_d00, lookUp));
	vec3f camera_ddv = cosFovy * normalize(cross(camera_ddu, camera_d00));
	camera_d00 -= 0.5f * camera_ddu;
	camera_d00 -= 0.5f * camera_ddv;
	owlRayGenSet3f(nanoContext.rayGen, "camera.dir00", reinterpret_cast<const owl3f&>(camera_d00));
	owlRayGenSet3f(nanoContext.rayGen, "camera.dirDu", reinterpret_cast<const owl3f&>(camera_ddu));
	owlRayGenSet3f(nanoContext.rayGen, "camera.dirDv", reinterpret_cast<const owl3f&>(camera_ddv));

	owlBuildSBT(nanoContext.context);

	owlLaunch2D(nanoContext.rayGen, view.colorRt.extent.width, view.colorRt.extent.height, nanoContext.lp);
	//owlRayGenLaunch2D(nanoContext.rayGen, view.colorRt.extent.width, view.colorRt.extent.height);

	{
		for (auto i = 0; i < view.colorRt.extent.depth; i++)
		{
			{
				const auto error = cudaDestroySurfaceObject(cudaSurfaceObjects[i]);
			}
		}
		{
			const auto error = cudaGraphicsUnmapResources(1, const_cast<cudaGraphicsResource_t*>(&view.colorRt.target));
		}
	}

	
	auto signalParams = cudaExternalSemaphoreSignalParams{};
	signalParams.flags = 0;
	signalParams.params.fence.value = 0;
	cudaSignalExternalSemaphoresAsync(&initializationInfo_.waitSemaphore, &signalParams, 1);
}

auto NanoRenderer::onInitialize() -> void
{
	bar();
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
