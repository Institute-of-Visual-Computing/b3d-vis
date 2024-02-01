#include "NanoRenderer.h"
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

#include <NanoCutterParser.h>

extern "C" char NanoRenderer_ptx[];
extern "C" uint8_t NanoRenderer_optixir[];
extern "C" uint32_t NanoRenderer_optixir_length;


using namespace b3d::renderer;
using namespace b3d::renderer::nano;
using namespace owl::common;


namespace
{
	struct GuiData
	{
		struct BackgroundColorPalette
		{
			std::array<float, 3> color1{ 0.572f, 0.100f, 0.750f };
			std::array<float, 3> color2{ 0.0f, 0.3f, 0.3f };
		};
		BackgroundColorPalette rtBackgroundColorPalette;
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

	[[maybe_unused]] auto getOptixTransform(const nanovdb::GridHandle<>& grid, float transform[]) -> void
	{
		// Extract the index-to-world-space affine transform from the Grid and convert
		// to 3x4 row-major matrix for Optix.
		auto* gridHandle = grid.grid<float>();
		const auto& map = gridHandle->map();
		transform[0] = map.mMatF[0];
		transform[1] = map.mMatF[1];
		transform[2] = map.mMatF[2];
		transform[3] = map.mVecF[0];
		transform[4] = map.mMatF[3];
		transform[5] = map.mMatF[4];
		transform[6] = map.mMatF[5];
		transform[7] = map.mVecF[1];
		transform[8] = map.mMatF[6];
		transform[9] = map.mMatF[7];
		transform[10] = map.mMatF[8];
		transform[11] = map.mVecF[2];
	}


	auto createVolume() -> NanoVdbVolume
	{
		const auto testFile = std::filesystem::path{ "D:/datacubes/n4565_cut/funny.nvdb" };
		//const auto testFile = std::filesystem::path{ "D:/datacubes/n4565_cut/nano_level_0_224_257_177.nvdb" };
		//const auto testFile = std::filesystem::path{ "D:/datacubes/ska/40gb/sky_ldev_v2.nvdb" };
		
		
		assert(std::filesystem::exists(testFile));
		// owlInstanceGroupSetTransform
		auto volume = NanoVdbVolume{};
		//auto gridVolume = nanovdb::createFogVolumeTorus();
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
		//const auto dim = gridVolume.gridMetaData()->worldBBox().dim();
		//map.set(0.1, nanovdb::Vec3{-dim[0]*0.5,-dim[1]*0.5,-dim[2]*0.5 });

		volume.transform = AffineSpace3f{ orientation, position }; // AffineSpace3f{ orientation, position };

		{
			const auto& box = gridVolume.gridMetaData()->worldBBox();
			const auto min = owl::vec3f{ static_cast<float>(box.min()[0]), static_cast<float>(box.min()[1]),
										 static_cast<float>(box.min()[2]) };
			const auto max = owl::vec3f{ static_cast<float>(box.max()[0]), static_cast<float>(box.max()[1]),
										 static_cast<float>(box.max()[2]) };

			// volume.worldAabb = owl::box3f{ map.applyMapF(min), map.applyMapF(max) };
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

	


	std::filesystem::path b3dFilePath{};
} // namespace


auto NanoRenderer::prepareGeometry() -> void
{
	const auto context = owlContextCreate(nullptr, 1);

	nanoContext_.context = context;

	const auto module = owlModuleCreate(context, NanoRenderer_ptx);

	const auto optixirModule = owlModuleCreateFromIR(context, NanoRenderer_optixir, NanoRenderer_optixir_length);

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

	// const auto volume = createVolume();

	nanoVdbVolume = unique_volume_ptr(new NanoVdbVolume());
	*nanoVdbVolume.get() = createVolume();

	const auto geometryGroup = owlUserGeomGroupCreate(context, 1, &geometry);
	nanoContext_.worldGeometryGroup = owlInstanceGroupCreate(context, 1, &geometryGroup, nullptr, nullptr,
															 OWL_MATRIX_FORMAT_OWL, OPTIX_BUILD_FLAG_ALLOW_UPDATE);

	// auto t = owl::affine3f::scale(owl::vec3f(.01f,.01f,.01f));
	/*float a[12];
	getOptixTransform(nanoVdbVolume->, a);
	owlInstanceGroupSetTransform(world, 0, (const float*)&a,
							   OWL_MATRIX_FORMAT_OWL);*/


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
		};

		nanoContext_.launchParams =
			owlParamsCreate(nanoContext_.context, sizeof(LaunchParams), launchParamsVarsWithStruct.data(),
							launchParamsVarsWithStruct.size());
	}


	// owlBuildProgramsDebug(context);
	owlBuildPrograms(context);
	owlBuildPipeline(context);
	// owlBuildPrograms(context)
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

	owlInstanceGroupSetTransform(nanoContext_.worldGeometryGroup, 0, reinterpret_cast<const float*>(&trs_));
	owlGroupRefitAccel(nanoContext_.worldGeometryGroup);

	/*auto rsInv = trs_.l.inverse();
	auto d = std::array{ trs_.l.vx, trs_.l.vy, trs_.l.vz };
	auto dInv = std::array{ rsInv.vx, rsInv.vy, rsInv.vz };
	currentMap_.set(d.data(), dInv.data(), trs_.p);*/

	{
		// TODO: need OBB, AABB !!!!!!!!!!!!!!!!!!!!!!!!
		// pass Map to cuda and apply
		debugDraw().drawBox(trs_.p, nanoVdbVolume->indexBox.size(),
							owl::vec3f(0.1f, 0.82f, 0.15f), trs_.l);
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
	//owlParamsSetRaw()



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
}

auto NanoRenderer::onDeinitialize() -> void
{
}

auto NanoRenderer::onGui() -> void
{
	ImGui::Begin("RT Settings");
	if(ImGui::Button("Reset Model Transform"))
	{
		rendererState_->worldMatTRS = AffineSpace3f{};
	}
	ImGui::SeparatorText("Data File (.b3d)");
	ImGui::InputText("##source", const_cast<char*>(b3dFilePath.string().c_str()), b3dFilePath.string().size(),
					 ImGuiInputTextFlags_ReadOnly);
	ImGui::SameLine();
	if (ImGui::Button("Select"))
	{
		ImGui::OpenPopup("FileSelectDialog");
	}
	ImGui::SameLine();
	if (ImGui::Button("Load"))
	{
		if (std::filesystem::exists(b3dFilePath))
		{
			const auto t = cutterParser::load(b3dFilePath.generic_string());

			const auto nanoFile = t.front().nanoVdbFile;
		}
		else
		{
			ImGui::TextColored(ImVec4{ 0.9f, 0.1f, 0.1f, 1.0f }, "Error: Can't load file!");
		}
	}
	ImGui::SeparatorText("Background Color Palette");
	ImGui::ColorEdit3("Color 1", guiData.rtBackgroundColorPalette.color1.data());
	ImGui::ColorEdit3("Color 2", guiData.rtBackgroundColorPalette.color2.data());

	ImGui::SeparatorText("Timings");

	const auto timing = gpuTimers_.get("basic owl rt");

	static float values[100] = {};
	static auto valuesOffset = 0;

	values[valuesOffset] = timing;
	valuesOffset = (valuesOffset + 1) % IM_ARRAYSIZE(values);

	{
		auto average = 0.0f;
		for (auto n = 0; n < IM_ARRAYSIZE(values); n++)
			average += values[n];
		average /= static_cast<float>(IM_ARRAYSIZE(values));
		ImGui::PlotHistogram("##perfGraph", values, IM_ARRAYSIZE(values), valuesOffset,
							 std::format("avg {}", average).c_str(), 0.0f, 16.0f, ImVec2(0, 400.0f));
	}

	ImGui::Text("%1.3f", timing);

	debugInfo_.gizmoHelper->drawGizmo(rendererState_->worldMatTRS);

	static auto currentPath = std::filesystem::current_path();
	static auto selectedPath = std::filesystem::path{};

	const auto center = ImGui::GetMainViewport()->GetCenter();
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

	if (ImGui::BeginPopupModal("FileSelectDialog", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
	{
		constexpr auto roots = std::array{ "A:/", "B:/", "C:/", "D:/", "E:/", "F:/", "G:/", "H:/", "I:/" };

		for (auto i = 0; i < roots.size(); i++)
		{
			const auto root = std::filesystem::path{ roots[i] };
			if (is_directory(root))
			{
				ImGui::SameLine();
				if (ImGui::Button(roots[i]))
				{
					currentPath = root;
				}
			}
		}
		if (ImGui::BeginListBox("##dirs", ImVec2(ImGui::GetFontSize() * 40, ImGui::GetFontSize() * 16)))
		{
			if (ImGui::Selectable("...", false))
			{
				currentPath = currentPath.parent_path();
			}
			auto i = 0;
			for (auto& dir : std::filesystem::directory_iterator{ currentPath })
			{
				i++;
				const auto path = dir.path();
				if (is_directory(path))
				{
					if (ImGui::Selectable(dir.path().string().c_str(), false))
					{
						currentPath = path;
					}
				}
				if (path.has_extension() && path.extension() == ".b3d")
				{
					ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.1f, 0.9f, 0.1f, 1.0f));
					if (ImGui::Selectable(dir.path().string().c_str(), dir.path() == selectedPath))
					{
						selectedPath = dir.path();
					}
					ImGui::PopStyleColor();
				}
			}
			ImGui::EndListBox();
		}
		if (ImGui::Button("OK", ImVec2(120, 0)))
		{
			if (!selectedPath.empty() != 0)
			{
				b3dFilePath = selectedPath;
			}
			ImGui::CloseCurrentPopup();
		}
		ImGui::SetItemDefaultFocus();
		ImGui::SameLine();
		if (ImGui::Button("Cancel", ImVec2(120, 0)))
		{
			selectedPath.clear();
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}

	ImGui::End();
}
