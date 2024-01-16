#include "RendererBase.h"
#include "SimpleTrianglesRenderer.h"

// public owl node-graph API
#include "owl/owl.h"
// our device-side data structures
#include "deviceCode.h"
#include "imgui.h"
#include "owl/helper/cuda.h"

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

	struct GuiData
	{
		struct BackgroundColorPalette
		{
			std::array<float, 4> color1{ 0.572f, 0.100f, 0.750f, 1.0f };
			std::array<float, 4> color2{ 0.0f, 0.3f, 0.3f, 1.0f };
		};
		BackgroundColorPalette rtBackgroundColorPalette;

		struct CubeVolumeTransform
		{
			std::array<float, 3> position{ 0, 0, 0 };
			std::array<float, 3> scale{ 1, 1, 1 };
			std::array<float, 3> rotation{ 0, 0, 0 };
		};
		CubeVolumeTransform rtCubeVolumeTransform{};
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

}

auto SimpleTrianglesRenderer::setCubeVolumeTransform(NativeCube* nativeCube) -> void
{
	auto translate = affine3f::translate(nativeCube->position);
	auto scale = affine3f::scale(nativeCube->scale);
	AffineSpace3f rotate{ nativeCube->rotation };

	trs_ = translate * rotate * scale;
}

auto SimpleTrianglesRenderer::onRender(const View& view) -> void
{
	auto waitParams = cudaExternalSemaphoreWaitParams{};
	waitParams.flags = 0;
	waitParams.params.fence.value = view.fenceValue;
	cudaWaitExternalSemaphoresAsync(&initializationInfo_.signalSemaphore, &waitParams, 1);

	// map/create/set surface
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
	
	if (fbSize_.x != view.colorRt.extent.width || fbSize_.y != view.colorRt.extent.height)
	{
		fbSize_ = { static_cast<int32_t>(view.colorRt.extent.width), static_cast<int32_t>(view.colorRt.extent.height) };
		owlRayGenSet2i(rayGen_, "fbSize", fbSize_);
		sbtDirty = true;
	}

	if (sbtDirty)
	{
		owlBuildSBT(context_);
		sbtDirty = false;
	}

	owlInstanceGroupSetTransform(world_, 0, (const float*)&trs_);
	owlGroupRefitAccel(world_);

	{
		const auto rcd = createRayCameraData(view.cameras[0], view.colorRt.extent);
		owlParamsSetRaw(launchParameters_, "cameraData", &rcd);
		owlParamsSetRaw(launchParameters_, "surfacePointer", &cudaSurfaceObjects[0]);
		owlAsyncLaunch2D(rayGen_, view.colorRt.extent.width, view.colorRt.extent.height, launchParameters_);
	}

	if(view.mode == RenderMode::stereo && view.colorRt.extent.depth > 1)
	{
		const auto rcd = createRayCameraData(view.cameras[1], view.colorRt.extent);
		owlParamsSetRaw(launchParameters_, "cameraData", &rcd);
		owlParamsSetRaw(launchParameters_, "surfacePointer", &cudaSurfaceObjects[1]);
		owlAsyncLaunch2D(rayGen_, view.colorRt.extent.width, view.colorRt.extent.height, launchParameters_);
	}

	
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
}

auto SimpleTrianglesRenderer::onInitialize() -> void
{
	RendererBase::onInitialize();

	trs_ = affine3f::translate({ 0, 0, 0 }).scale({ 1, 1, 1 });

	// create a context on the first device:
	context_ = owlContextCreate(nullptr, 1);
	auto module = owlModuleCreate(context_, SimpleTrianglesDeviceCode_ptx);


	// ##################################################################
	// set up all the *GEOMETRY* graph we want to render
	// ##################################################################

	// -------------------------------------------------------
	// declare geometry type
	// -------------------------------------------------------
	OWLVarDecl trianglesGeomVars[] = { { "index", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, index) },
									   { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData, vertex) },
									   { "color", OWL_FLOAT4, OWL_OFFSETOF(TrianglesGeomData, color) } };
	OWLGeomType trianglesGeomType =
		owlGeomTypeCreate(context_, OWL_TRIANGLES, sizeof(TrianglesGeomData), trianglesGeomVars, 3);
	owlGeomTypeSetClosestHit(trianglesGeomType, 0, module, "TriangleMesh");

	// ##################################################################
	// set up all the *GEOMS* we want to run that code on
	// ##################################################################

	// ------------------------------------------------------------------
	// triangle mesh
	// ------------------------------------------------------------------
	OWLBuffer vertexBuffer = owlDeviceBufferCreate(context_, OWL_FLOAT3, NUM_VERTICES, vertices);
	OWLBuffer indexBuffer = owlDeviceBufferCreate(context_, OWL_INT3, NUM_INDICES, indices);

	OWLGeom trianglesGeom = owlGeomCreate(context_, trianglesGeomType);

	owlTrianglesSetVertices(trianglesGeom, vertexBuffer, NUM_VERTICES, sizeof(vec3f), 0);
	owlTrianglesSetIndices(trianglesGeom, indexBuffer, NUM_INDICES, sizeof(vec3i), 0);

	owlGeomSetBuffer(trianglesGeom, "vertex", vertexBuffer);
	owlGeomSetBuffer(trianglesGeom, "index", indexBuffer);
	owlGeomSet4f(trianglesGeom, "color", owl4f{ 0, 1, 0,1 });

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
		OWLVarDecl rayGenVars[] = { 
									{ "fbSize", OWL_INT2, OWL_OFFSETOF(RayGenData, fbSize) },
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
	ImGui::Begin("RT Settings");
	ImGui::SeparatorText("Background Color Palette");
	ImGui::ColorEdit3("Color 1", guiData.rtBackgroundColorPalette.color1.data());
	ImGui::ColorEdit3("Color 2", guiData.rtBackgroundColorPalette.color2.data());
	ImGui::SeparatorText("CubeVolume Transform");
	ImGui::DragFloat3("Position", guiData.rtCubeVolumeTransform.position.data(), 0.01f, -FLT_MAX, FLT_MAX);
	ImGui::DragFloat3("Scale", guiData.rtCubeVolumeTransform.scale.data(), 0.01f, 0.01f, 100.0f);
	ImGui::DragFloat3("Rotation", guiData.rtCubeVolumeTransform.rotation.data(), 0.01f, -FLT_MAX, FLT_MAX);
	if (ImGui::Button("Reset Transform", ImVec2(ImGui::GetContentRegionAvail().x, 0)))
	{
		guiData.rtCubeVolumeTransform = {};
	}

	ImGui::End();

	auto nt = NativeCube{
		{ guiData.rtCubeVolumeTransform.position[0], guiData.rtCubeVolumeTransform.position[1],
		  guiData.rtCubeVolumeTransform.position[2] },
		{ guiData.rtCubeVolumeTransform.scale[0], guiData.rtCubeVolumeTransform.scale[1],
		  guiData.rtCubeVolumeTransform.scale[2] },
		Quaternion3f{ guiData.rtCubeVolumeTransform.rotation[0], guiData.rtCubeVolumeTransform.rotation[1],
					  guiData.rtCubeVolumeTransform.rotation[2] },

	};
	this->setCubeVolumeTransform(&nt);

}
