#include "RendererBase.h"
#include "SimpleTrianglesRenderer.h"

// public owl node-graph API
#include "owl/owl.h"
// our device-side data structures
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
}

const int NUM_VERTICES = 8;
vec3f vertices[NUM_VERTICES] = {
	{ -1.f, -1.f, -1.f }, { +1.f, -1.f, -1.f }, { -1.f, +1.f, -1.f }, { +1.f, +1.f, -1.f },
	{ -1.f, -1.f, +1.f }, { +1.f, -1.f, +1.f }, { -1.f, +1.f, +1.f }, { +1.f, +1.f, +1.f }
};

const int NUM_INDICES = 12;
vec3i indices[NUM_INDICES] = { { 0, 1, 3 }, { 2, 3, 0 }, { 5, 7, 6 }, { 5, 6, 4 }, { 0, 4, 5 }, { 0, 5, 1 },
							   { 2, 3, 7 }, { 2, 7, 6 }, { 1, 5, 7 }, { 1, 7, 3 }, { 4, 0, 2 }, { 4, 2, 6 } };


// const vec2i fbSize(800,600);
const vec3f init_lookFrom(-4.f, +3.f, -2.f);
const vec3f init_lookAt(0.f, 0.f, 0.f);
const vec3f init_lookUp(0.f, 1.f, 0.f);
const float init_cosFovy = 0.66f;



auto SimpleTrianglesRenderer::onRender(const View& view) -> void
{
	auto waitParams = cudaExternalSemaphoreWaitParams{};
	waitParams.flags = 0;
	waitParams.params.fence.value = 1;
	cudaWaitExternalSemaphoresAsync(&initializationInfo_.signalSemaphore, &waitParams, 1);


	// map/create/set surface
	// TODO: class members
	std::array<cudaArray_t, 2> cudaArrays{};
	std::array<cudaSurfaceObject_t, 2> cudaSurfaceObjects{};

	auto cudaRet = cudaSuccess;
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


	owlBufferUpload(surfaceBuffer_, cudaSurfaceObjects.data(), 0, 1);

	owlRayGenSetBuffer(rayGen, "fbPtr", surfaceBuffer_);

	owl2i fbSize = {
		static_cast<int32_t>(view.colorRt.extent.width),
		static_cast<int32_t>(view.colorRt.extent.height)
	};
	owlRayGenSet2i(rayGen, "fbSize", fbSize);
	
	owlRayGenSet3f(rayGen, "camera.pos", (const owl3f&)view.cameras[0].origin);
	


	// ----------- compute variable values  ------------------
	const auto& camera = view.cameras.front();
	const auto origin = vec3f{ camera.origin.x, camera.origin.y, camera.origin.z };
	auto camera_d00 = camera.at - origin;
	const auto wlen = length(camera_d00);
	owlRayGenSet3f(rayGen, "camera.pos", reinterpret_cast<const owl3f&>(origin));
	const auto aspect = view.colorRt.extent.width / static_cast<float>(view.colorRt.extent.height);
	const auto vlen = wlen * std::cosf(0.5f * camera.FoV);
	const auto camera_ddu = vlen * aspect * normalize(cross(camera_d00, camera.up));
	const auto camera_ddv = vlen * normalize(cross(camera_ddu, camera_d00));
	/*camera_d00 -= 0.5f * camera_ddu;
	camera_d00 -= 0.5f * camera_ddv;*/

	const auto vz = -normalize(camera.at - origin);
	const auto vx = normalize(cross(camera.up, vz));
	const auto vy = normalize(cross(vz, vx));
	const auto focalDistance = length(camera.at - origin);
	const float minFocalDistance = max(computeStableEpsilon(origin), computeStableEpsilon(vx));

	/*
	  tan(fov/2) = (height/2) / dist
	  -> height = 2*tan(fov/2)*dist
	*/
	float screen_height = 2.f * tanf(camera.FoV / 2.f) * max(minFocalDistance, focalDistance);
	const auto vertical = screen_height * vy;
	const auto horizontal = screen_height * aspect * vx;
	const auto lower_left = -max(minFocalDistance, focalDistance) * vz - 0.5f * vertical - 0.5f * horizontal;


	owlRayGenSet3f(rayGen, "camera.dir_00", reinterpret_cast<const owl3f&>(lower_left));
	owlRayGenSet3f(rayGen, "camera.dir_du", reinterpret_cast<const owl3f&>(horizontal));
	owlRayGenSet3f(rayGen, "camera.dir_dv", reinterpret_cast<const owl3f&>(vertical));

	owlBuildSBT(context);
	sbtDirty = true;
	owlRayGenLaunch2D(rayGen, view.colorRt.extent.width, view.colorRt.extent.height);

	// Destroy and unmap
	{
		for (auto i = 0; i < view.colorRt.extent.depth; i++)
		{
			cudaRet = cudaDestroySurfaceObject(cudaSurfaceObjects[i]);
		}
		cudaRet = cudaGraphicsUnmapResources(1, const_cast<cudaGraphicsResource_t*>(&view.colorRt.target));
	}

	constexpr std::array signalParams = { cudaExternalSemaphoreSignalParams{ { { 1 } }, 0 },
										  cudaExternalSemaphoreSignalParams{ { { 0 } }, 0 } };
	cudaSignalExternalSemaphoresAsync(&initializationInfo_.waitSemaphore, signalParams.data(), 2);
}

auto SimpleTrianglesRenderer::onInitialize() -> void
{
	RendererBase::onInitialize();
	// create a context on the first device:
	context = owlContextCreate(nullptr, 1);
	OWLModule module = owlModuleCreate(context, SimpleTrianglesDeviceCode_ptx);

	// ##################################################################
	// set up all the *GEOMETRY* graph we want to render
	// ##################################################################

	// -------------------------------------------------------
	// declare geometry type
	// -------------------------------------------------------
	OWLVarDecl trianglesGeomVars[] = { { "index", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index) },
									   { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex) },
									   { "color", OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData, color) } };
	OWLGeomType trianglesGeomType =
		owlGeomTypeCreate(context, OWL_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars, 3);
	owlGeomTypeSetClosestHit(trianglesGeomType, 0, module, "TriangleMesh");

	// ##################################################################
	// set up all the *GEOMS* we want to run that code on
	// ##################################################################


	// ------------------------------------------------------------------
	// triangle mesh
	// ------------------------------------------------------------------
	OWLBuffer vertexBuffer = owlDeviceBufferCreate(context, OWL_FLOAT3, NUM_VERTICES, vertices);
	OWLBuffer indexBuffer = owlDeviceBufferCreate(context, OWL_INT3, NUM_INDICES, indices);
	// OWLBuffer frameBuffer
	//   = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

	OWLGeom trianglesGeom = owlGeomCreate(context, trianglesGeomType);

	owlTrianglesSetVertices(trianglesGeom, vertexBuffer, NUM_VERTICES, sizeof(vec3f), 0);
	owlTrianglesSetIndices(trianglesGeom, indexBuffer, NUM_INDICES, sizeof(vec3i), 0);

	owlGeomSetBuffer(trianglesGeom, "vertex", vertexBuffer);
	owlGeomSetBuffer(trianglesGeom, "index", indexBuffer);
	owlGeomSet3f(trianglesGeom, "color", owl3f{ 0, 1, 0 });

	// ------------------------------------------------------------------
	// the group/accel for that mesh
	// ------------------------------------------------------------------
	OWLGroup trianglesGroup = owlTrianglesGeomGroupCreate(context, 1, &trianglesGeom);
	owlGroupBuildAccel(trianglesGroup);
	OWLGroup world = owlInstanceGroupCreate(context, 1, &trianglesGroup);
	owlGroupBuildAccel(world);


	// ##################################################################
	// set miss and raygen program required for SBT
	// ##################################################################

	// -------------------------------------------------------
	// set up miss prog
	// -------------------------------------------------------
	OWLVarDecl missProgVars[] = { { "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color0) },
								  { "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData, color1) },
								  { /* sentinel to mark end of list */ } };
	// ----------- create object  ----------------------------
	OWLMissProg missProg = owlMissProgCreate(context, module, "miss", sizeof(MissProgData), missProgVars, -1);

	// ----------- set variables  ----------------------------
	owlMissProgSet3f(missProg, "color0", owl3f{ .8f, 0.f, 0.f });
	owlMissProgSet3f(missProg, "color1", owl3f{ .8f, .8f, .8f });

	// -------------------------------------------------------
	// set up ray gen program
	// -------------------------------------------------------
	OWLVarDecl rayGenVars[] = { { "fbPtr", OWL_BUFPTR, OWL_OFFSETOF(RayGenData, fbPtr) },
								// { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
								{ "fbSize", OWL_INT2, OWL_OFFSETOF(RayGenData, fbSize) },
								{ "world", OWL_GROUP, OWL_OFFSETOF(RayGenData, world) },
								{ "camera.pos", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.pos) },
								{ "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_00) },
								{ "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_du) },
								{ "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData, camera.dir_dv) },
								{ /* sentinel to mark end of list */ } };


	// Create Buffer for Surfaceobject
	// TODO: Does not work if we have 2 colorRT in view
	surfaceBuffer_ = owlDeviceBufferCreate(context, OWL_USER_TYPE(cudaSurfaceObject_t), 1, nullptr);


	// ----------- create object  ----------------------------
	rayGen = owlRayGenCreate(context, module, "simpleRayGen", sizeof(RayGenData), rayGenVars, -1);
	/* camera and frame buffer get set in resiez() and cameraChanged() */
	owlRayGenSetGroup(rayGen, "world", world);

	// ##################################################################
	// build *SBT* required to trace the groups
	// ##################################################################

	owlBuildPrograms(context);
	owlBuildPipeline(context);
	owlBuildSBT(context);
}
