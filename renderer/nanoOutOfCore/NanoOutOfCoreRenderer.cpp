#include "NanoOutOfCoreRenderer.h"
#include <nanovdb/NanoVDB.h>
#include <owl/helper/cuda.h>
#include "cuda_runtime.h"

#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/Primitives.h>

#include <nanovdb/util/IO.h>

#include <cuda.h>

#include <filesystem>

#include "SharedStructs.h"
#include "owl/owl_host.h"

#include <format>

#include "DebugDrawListBase.h"

#include <imgui.h>
#include <imgui_internal.h>

#include <NanoCutterParser.h>

extern "C" char NanoOutOfCoreRenderer_ptx[];
extern "C" uint8_t NanoOutOfCoreRenderer_optixir[];
extern "C" uint32_t NanoOutOfCoreRenderer_optixir_length;


using namespace b3d::renderer;
using namespace b3d::renderer::nano;
using namespace owl::common;


namespace
{
	vec3f spectral_jet(float x)
	{
		vec3f c;
		if (x < 0.25)
			c = vec3f(0.0, 4.0 * x, 1.0);
		else if (x < 0.5)
			c = vec3f(0.0, 1.0, 1.0 + 4.0 * (0.25 - x));
		else if (x < 0.75)
			c = vec3f(4.0 * (x - 0.5), 1.0, 0.0);
		else
			c = vec3f(1.0, 1.0 + 4.0 * (0.75 - x), 0.0);
		return clamp(c, vec3f(0.0), vec3f(1.0));
	}

	struct GuiData
	{
		struct BackgroundColorPalette
		{
			std::array<float, 3> color1{ 0.572f, 0.100f, 0.750f };
			std::array<float, 3> color2{ 0.0f, 0.3f, 0.3f };
		};
		BackgroundColorPalette rtBackgroundColorPalette;
		bool fillBox{ false };
		std::array<float, 3> fillColor{ 0.8f, 0.3f, 0.2f };
		std::array<float, 3> color{ 1.0f, 1.0f, 1.0f };
	};

	GuiData guiData{};

	struct NanoVdbVolumeDeleter
	{
		auto operator()(const NanoVdbVolume* volume) const noexcept -> void
		{
			OWL_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(volume->grid)));
			delete volume;
		}
	};

	using unique_volume_ptr = std::unique_ptr<NanoVdbVolume, NanoVdbVolumeDeleter>;

	unique_volume_ptr nanoVdbVolume;

	auto createVolume() -> NanoVdbVolume
	{
		const auto testFile = std::filesystem::path{ "D:/datacubes/n4565_cut/funny.nvdb" };
		// const auto testFile =
		std::filesystem::path{ "D:/datacubes/n4565_cut/filtered_level_0_224_257_177_id_7_upscale.fits.nvdb" };
		// const auto testFile = std::filesystem::path{ "D:/datacubes/n4565_cut/nano_level_0_224_257_177.nvdb" };
		// const auto testFile = std::filesystem::path{ "D:/datacubes/ska/40gb/sky_ldev_v2.nvdb" };


		assert(std::filesystem::exists(testFile));
		// owlInstanceGroupSetTransform
		auto volume = NanoVdbVolume{};
		// auto gridVolume = nanovdb::createFogVolumeTorus();
		const auto gridVolume = nanovdb::io::readGrid(testFile.string());
		OWL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&volume.grid), gridVolume.size()));
		OWL_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(volume.grid), gridVolume.data(), gridVolume.size(),
								  cudaMemcpyHostToDevice));

		const auto gridHandle = gridVolume.grid<float>();
		const auto& map = gridHandle->mMap;
		const auto orientation =
			owl::LinearSpace3f{ map.mMatF[0], map.mMatF[1], map.mMatF[2], map.mMatF[3], map.mMatF[4],
								map.mMatF[5], map.mMatF[6], map.mMatF[7], map.mMatF[8] };
		const auto position = vec3f{ 0.0, 0.0, 0.0 };

		volume.transform = AffineSpace3f{ orientation, position };

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

	auto orientedBoxToBox(const owl::box3f& box, const LinearSpace3f& orientation) -> owl::box3f
	{
		const auto midPoint = owl::vec3f{ box.center().x, box.center().y, box.center().z };
		const auto extent = owl::vec3f{ box.size().x, box.size().y, box.size().z };

		/*
		 *	  p4----p5
		 *	 / |   / |
		 *	p0----p1 |
		 *	|  p6--|-p7
		 *	| /    |/
		 *	p2----p3
		 */
		const auto p0 = orientation * (midPoint + 0.5f * owl::vec3f(-1.0, -1.0, 1.0) * extent);
		const auto p1 = orientation * (midPoint + 0.5f * owl::vec3f(1.0, -1.0, 1.0) * extent);
		const auto p2 = orientation * (midPoint + 0.5f * owl::vec3f(-1.0, -1.0, -1.0) * extent);
		const auto p3 = orientation * (midPoint + 0.5f * owl::vec3f(1.0, -1.0, -1.0) * extent);
		const auto p4 = orientation * (midPoint + 0.5f * owl::vec3f(-1.0, 1.0, 1.0) * extent);
		const auto p5 = orientation * (midPoint + 0.5f * owl::vec3f(1.0, 1.0, 1.0) * extent);
		const auto p6 = orientation * (midPoint + 0.5f * owl::vec3f(-1.0, 1.0, -1.0) * extent);
		const auto p7 = orientation * (midPoint + 0.5f * owl::vec3f(1.0, 1.0, -1.0) * extent);

		auto newBox = owl::box3f{};
		newBox.extend(p0);
		newBox.extend(p1);
		newBox.extend(p2);
		newBox.extend(p3);
		newBox.extend(p4);
		newBox.extend(p5);
		newBox.extend(p6);
		newBox.extend(p7);

		return newBox;
	}
} // namespace


auto NanoRenderer::prepareGeometry() -> void
{
	const auto context = owlContextCreate(nullptr, 1);

	nanoContext_.context = context;

	const auto module = owlModuleCreate(context, NanoOutOfCoreRenderer_ptx);

	const auto optixirModule =
		owlModuleCreateFromIR(context, NanoOutOfCoreRenderer_optixir, NanoOutOfCoreRenderer_optixir_length);

	[[maybe_unused]] const auto volumeGeometryVars =
		std::array{ OWLVarDecl{ "indexBox", OWL_FLOAT3, OWL_OFFSETOF(NanoVdbVolume, indexBox) },
					OWLVarDecl{ "worldAabb", OWL_FLOAT3, OWL_OFFSETOF(NanoVdbVolume, worldAabb) },
					OWLVarDecl{ "indexBox", OWL_AFFINE3F, OWL_OFFSETOF(NanoVdbVolume, transform) },
					OWLVarDecl{ "grid", OWL_BUFFER_POINTER, OWL_OFFSETOF(NanoVdbVolume, grid) } };

	const auto geometryVars =
		std::array{ OWLVarDecl{ "volume", OWL_USER_TYPE(NanoVdbVolume), OWL_OFFSETOF(GeometryData, volume) } };

	const auto geometryType =
		owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(GeometryData), geometryVars.data(), geometryVars.size());

	const auto rayGenerationVars =
		std::array{ OWLVarDecl{ "frameBufferSize", OWL_INT2, OWL_OFFSETOF(RayGenerationData, frameBufferSize) },
					OWLVarDecl{ "world", OWL_GROUP, OWL_OFFSETOF(RayGenerationData, world) } };

	const auto rayGen = owlRayGenCreate(context, optixirModule, "rayGeneration", sizeof(RayGenerationData),
										rayGenerationVars.data(), rayGenerationVars.size());

	nanoContext_.rayGen = rayGen;

	auto geometry = owlGeomCreate(context, geometryType);

	nanoVdbVolume = unique_volume_ptr(new NanoVdbVolume());
	*nanoVdbVolume.get() = createVolume();

	const auto geometryGroup = owlUserGeomGroupCreate(context, 1, &geometry);
	nanoContext_.worldGeometryGroup = owlInstanceGroupCreate(context, 1, &geometryGroup, nullptr, nullptr,
															 OWL_MATRIX_FORMAT_OWL, OPTIX_BUILD_FLAG_ALLOW_UPDATE);

	owlGeomSetRaw(geometry, "volume", nanoVdbVolume.get());

	owlGeomTypeSetBoundsProg(geometryType, module, "volumeBounds");
	owlBuildPrograms(context);

	owlGeomSetPrimCount(geometry, 1);

	owlGroupBuildAccel(geometryGroup);
	owlGroupBuildAccel(nanoContext_.worldGeometryGroup);

	owlRayGenSetGroup(rayGen, "world", nanoContext_.worldGeometryGroup);

	owlGeomTypeSetIntersectProg(geometryType, 0, optixirModule, "nano_intersection");
	owlGeomTypeSetClosestHit(geometryType, 0, optixirModule, "nano_closestHit");


	const auto missProgramVars =
		std::array{ OWLVarDecl{ "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgramData, color0) },
					OWLVarDecl{ "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgramData, color1) } };

	// ----------- create object  ----------------------------
	nanoContext_.missProgram = owlMissProgCreate(context, optixirModule, "miss", sizeof(MissProgramData),
												 missProgramVars.data(), missProgramVars.size());

	// ----------- set variables  ----------------------------
	owlMissProgSet3f(nanoContext_.missProgram, "color0", owl3f{ .8f, 0.f, 0.f });
	owlMissProgSet3f(nanoContext_.missProgram, "color1", owl3f{ .8f, .8f, .8f });


	{
		const auto launchParamsVarsWithStruct = std::array{
			OWLVarDecl{ "cameraData", OWL_USER_TYPE(RayCameraData), OWL_OFFSETOF(LaunchParams, cameraData) },
			OWLVarDecl{ "surfacePointer", OWL_USER_TYPE(cudaSurfaceObject_t),
						OWL_OFFSETOF(LaunchParams, surfacePointer) },
			OWLVarDecl{ "bg.color0", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, bg.color0) },
			OWLVarDecl{ "bg.color1", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, bg.color1) },
			OWLVarDecl{ "bg.fillBox", OWL_BOOL, OWL_OFFSETOF(LaunchParams, bg.fillBox) },
			OWLVarDecl{ "bg.fillColor", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, bg.fillColor) },
			OWLVarDecl{ "color", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, color) },

		};

		nanoContext_.launchParams =
			owlParamsCreate(nanoContext_.context, sizeof(LaunchParams), launchParamsVarsWithStruct.data(),
							launchParamsVarsWithStruct.size());
	}

	owlBuildPrograms(context);
	owlBuildPipeline(context);
	owlBuildSBT(context);
}

auto NanoRenderer::onRender(const View& view) -> void
{
	gpuTimers_.nextFrame();

	auto waitParams = cudaExternalSemaphoreWaitParams{};
	waitParams.flags = 0;
	waitParams.params.fence.value = view.fenceValue;
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
	trs_ = rendererState_->worldMatTRS;


	const auto volumeTranslate = AffineSpace3f::translate(-nanoVdbVolume->indexBox.size() * 0.5f);
	const auto transform = trs_ * volumeTranslate;
	owlInstanceGroupSetTransform(nanoContext_.worldGeometryGroup, 0, reinterpret_cast<const float*>(&transform));
	owlGroupRefitAccel(nanoContext_.worldGeometryGroup);


	{
		/*debugDraw().drawBox(trs_.p, nanoVdbVolume->indexBox.size(), owl::vec4f(0.1f, 0.82f, 0.15f, 1.0f), trs_.l);

		const auto aabbSize = orientedBoxToBox(nanoVdbVolume->indexBox, trs_.l).size();
		debugDraw().drawBox(trs_.p, aabbSize, owl::vec4f(0.9f, 0.4f, 0.2f, 0.4f));*/
	}


	if (dataSet_.has_value())
	{
		const auto& set = dataSet_.value();
		const auto box = box3f{ vec3f{ set.setBox.min[0], set.setBox.min[1], set.setBox.min[2] },
								vec3f{ set.setBox.max[0], set.setBox.max[1], set.setBox.max[2] } };


		debugDraw().drawBox(trs_.p, vec3f{ 0.0, 0.0, 0.0 }, box.size(), owl::vec4f(0.1f, 0.82f, 0.15f, 1.0f), trs_.l);

		for (const auto& cluster : set.clusters)
		{

			struct Node
			{
				cutterParser::TreeNode node;
				int level{};
			};

			std::stack<Node> p;
			p.push(Node{ cluster.accelerationStructureRoot, 0 });
			if (visibleLevelRange[0] <= 0 && 0 <= visibleLevelRange[1])
			{
				debugDraw().drawBox(trs_.p, trs_.p, box.size(), owl::vec4f(0.1f, 0.82f, 0.15f, 1.0f), trs_.l);
			}

			while (!p.empty())
			{
				const auto node = p.top();
				p.pop();


				const auto aabbBox =
					box3f{ vec3f{ node.node.aabb.min[0], node.node.aabb.min[1], node.node.aabb.min[2] } - box.center(),
						   vec3f{ node.node.aabb.max[0], node.node.aabb.max[1], node.node.aabb.max[2] } -
							   box.center() };
				if (visibleLevelRange[0] <= node.level && node.level <= visibleLevelRange[1])
				{
					debugDraw().drawBox(trs_.p, aabbBox.center(), aabbBox.size(),
										owl::vec4f(spectral_jet(node.level / 10.0f), 1.0f), trs_.l);
				}
				for (const auto& child : node.node.children)
				{
					p.push({ child, node.level + 1 });
				}
			}
		}
	}


	owlMissProgSet3f(nanoContext_.missProgram, "color0",
					 owl3f{ guiData.rtBackgroundColorPalette.color1[0], guiData.rtBackgroundColorPalette.color1[1],
							guiData.rtBackgroundColorPalette.color1[2] });
	owlMissProgSet3f(nanoContext_.missProgram, "color1",
					 owl3f{ guiData.rtBackgroundColorPalette.color2[0], guiData.rtBackgroundColorPalette.color2[1],
							guiData.rtBackgroundColorPalette.color2[2] });


	const auto fbSize =
		owl2i{ static_cast<int32_t>(view.colorRt.extent.width), static_cast<int32_t>(view.colorRt.extent.height) };
	owlRayGenSet2i(nanoContext_.rayGen, "frameBufferSize", fbSize);

	owlBuildSBT(nanoContext_.context);

	auto rcd = RayCameraData{};
	if (view.cameras[0].directionsAvailable)
	{
		rcd = { view.cameras[0].origin, view.cameras[0].dir00, view.cameras[0].dirDu, view.cameras[0].dirDv };
	}
	else
	{
		rcd = createRayCameraData(view.cameras[0], view.colorRt.extent);
	}

	owlParamsSetRaw(nanoContext_.launchParams, "cameraData", &rcd);
	owlParamsSetRaw(nanoContext_.launchParams, "surfacePointer", &cudaSurfaceObjects[0]);
	owlParamsSet3f(nanoContext_.launchParams, "bg.color0",
				   owl3f{ guiData.rtBackgroundColorPalette.color1[0], guiData.rtBackgroundColorPalette.color1[1],
						  guiData.rtBackgroundColorPalette.color1[2] });
	owlParamsSet3f(nanoContext_.launchParams, "bg.color1",
				   owl3f{ guiData.rtBackgroundColorPalette.color2[0], guiData.rtBackgroundColorPalette.color2[1],
						  guiData.rtBackgroundColorPalette.color2[2] });
	owlParamsSet1b(nanoContext_.launchParams, "bg.fillBox", guiData.fillBox);
	owlParamsSet3f(nanoContext_.launchParams, "bg.fillColor",
				   owl3f{ guiData.fillColor[0], guiData.fillColor[1], guiData.fillColor[2] });
	owlParamsSet3f(nanoContext_.launchParams, "color", owl3f{ guiData.color[0], guiData.color[1], guiData.color[2] });

	constexpr auto deviceId = 0;
	const auto stream = owlParamsGetCudaStream(nanoContext_.launchParams, deviceId);

	const auto record = gpuTimers_.record("basic owl rt", stream);

	record.start();

	owlAsyncLaunch2D(nanoContext_.rayGen, view.colorRt.extent.width, view.colorRt.extent.height,
					 nanoContext_.launchParams);
	record.stop();

	{
		for (auto i = 0; i < view.colorRt.extent.depth; i++)
		{
			OWL_CUDA_CHECK(cudaDestroySurfaceObject(cudaSurfaceObjects[i]));
		}
		OWL_CUDA_CHECK(cudaGraphicsUnmapResources(1, const_cast<cudaGraphicsResource_t*>(&view.colorRt.target)));
	}

	auto signalParams = cudaExternalSemaphoreSignalParams{};
	signalParams.flags = 0;
	signalParams.params.fence.value = view.fenceValue;
	cudaSignalExternalSemaphoresAsync(&initializationInfo_.waitSemaphore, &signalParams, 1);
}

auto NanoRenderer::onInitialize() -> void
{
	RendererBase::onInitialize();
	prepareGeometry();
	dataSet_ = cutterParser::load(std::filesystem::path{ "D:/datacubes/n4565_cut_2/project.b3d" });
}

auto NanoRenderer::onDeinitialize() -> void
{
}

auto NanoRenderer::onGui() -> void
{
	ImGui::Begin("RT Settings");
	if (ImGui::Button("Reset Model Transform"))
	{
		rendererState_->worldMatTRS = AffineSpace3f{};
	}
	ImGui::SameLine();
	if (ImGui::Button("Normalize Scaling"))
	{
		const auto scale = 1.0f / vec3f{ 10.0, 10.0, 10.0 };
		rendererState_->worldMatTRS *= AffineSpace3f::scale(scale);
	}
	ImGui::SeparatorText("Data File (.b3d)");

	const auto b3dFilePath = openFileDialog_.getSelectedItems().empty() ? std::filesystem::path{} :
																		  openFileDialog_.getSelectedItems().front();

	ImGui::InputText("##source", const_cast<char*>(b3dFilePath.string().c_str()), b3dFilePath.string().size(),
					 ImGuiInputTextFlags_ReadOnly);
	ImGui::SameLine();
	if (ImGui::Button("Select"))
	{
		openFileDialog_.open();
	}
	ImGui::SameLine();
	if (ImGui::Button("Load"))
	{
		if (std::filesystem::exists(b3dFilePath))
		{
			dataSet_ = cutterParser::load(b3dFilePath.generic_string());
		}
		else
		{
			ImGui::TextColored(ImVec4{ 0.9f, 0.1f, 0.1f, 1.0f }, "Error: Can't load file!");
		}
	}
	ImGui::ColorEdit3("Cloud Color", guiData.color.data());
	ImGui::SeparatorText("Background Color Palette");
	ImGui::ColorEdit3("Color 1", guiData.rtBackgroundColorPalette.color1.data());
	ImGui::ColorEdit3("Color 2", guiData.rtBackgroundColorPalette.color2.data());

	ImGui::Checkbox("Fill Box", &guiData.fillBox);

	if (guiData.fillBox)
	{
		ImGui::ColorEdit3("Fill Color", guiData.fillColor.data());
	}

	ImGui::DragIntRange2("Select Visible Level Range", &visibleLevelRange[0], &visibleLevelRange[1], 0.05, 0, 10);

	ImGui::SeparatorText("Timings");

	const auto timing = gpuTimers_.get("basic owl rt");

	static float values[100] = {};
	static auto valuesOffset = 0;

	values[valuesOffset] = timing;
	valuesOffset = (valuesOffset + 1) % IM_ARRAYSIZE(values);

	{
		auto average = 0.0f;
		for (auto n = 0; n < IM_ARRAYSIZE(values); n++)
		{
			average += values[n];
		}
		average /= static_cast<float>(IM_ARRAYSIZE(values));
		ImGui::PlotHistogram("##perfGraph", values, IM_ARRAYSIZE(values), valuesOffset,
							 std::format("avg {:3.2f} ms", average).c_str(), 0.0f, 16.0f, ImVec2(0, 400.0f));
	}

	ImGui::Text("%1.3f", timing);

	debugInfo_.gizmoHelper->drawGizmo(rendererState_->worldMatTRS);

	openFileDialog_.gui();


	ImGui::End();
}
