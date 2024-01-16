#include "NanoRenderer.h"
#include <Logging.h>
#include <nanovdb/NanoVDB.h>
#include <owl/helper/cuda.h>
#include "cuda_runtime.h"

#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/GridBuilder.h>
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

		OWLMissProg missProgram;
	};

	NanoContext nanoContext{};

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

	void getOptixTransform(const nanovdb::GridHandle<>& grid, float transform[])
	{
		// Extract the index-to-world-space affine transform from the Grid and convert
		// to 3x4 row-major matrix for Optix.
		auto* grid_handle = grid.grid<float>();
		const nanovdb::Map& map = grid_handle->map();
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
		assert(std::filesystem::exists(testFile));
		// owlInstanceGroupSetTransform
		auto volume = NanoVdbVolume{};
		// const auto gridVolume = nanovdb::createFogVolumeTorus();
		const auto gridVolume = nanovdb::io::readGrid(testFile.string());
		OWL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&volume.grid), gridVolume.size()));
		OWL_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(volume.grid), gridVolume.data(), gridVolume.size(),
								  cudaMemcpyHostToDevice));

		const auto gridHandle = gridVolume.grid<float>();
		const auto& map = gridHandle->map();

		const auto orientation =
			owl::LinearSpace3f{ map.mMatF[0], map.mMatF[1], map.mMatF[2], map.mMatF[3], map.mMatF[4],
								map.mMatF[5], map.mMatF[6], map.mMatF[7], map.mMatF[8] };
		const auto position = vec3f{ 0.0, 0.0, 0.0 };

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

		const auto geometryVars =
			std::array{ OWLVarDecl{ "volume", OWL_USER_TYPE(NanoVdbVolume), OWL_OFFSETOF(GeometryData, volume) } };

		const auto geometryType =
			owlGeomTypeCreate(context, OWL_GEOM_USER, sizeof(GeometryData), geometryVars.data(), geometryVars.size());

		const auto rayGenerationVars =
			std::array{ OWLVarDecl{ "surfacePointers", OWL_USER_TYPE(cudaSurfaceObject_t[2]),OWL_OFFSETOF(RayGenerationData, surfacePointers) },
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

		// const auto volume = createVolume();

		nanoVdbVolume = unique_volume_ptr(new NanoVdbVolume());
		*nanoVdbVolume.get() = createVolume();

		const auto geometryGroup = owlUserGeomGroupCreate(context, 1, &geometry);
		const auto world = owlInstanceGroupCreate(context, 1, &geometryGroup, nullptr, nullptr, OWL_MATRIX_FORMAT_OWL,
												  OPTIX_BUILD_FLAG_ALLOW_UPDATE);

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
		owlGroupBuildAccel(world);

		owlRayGenSetGroup(rayGen, "world", world);

		owlGeomTypeSetIntersectProg(geometryType, 0, module, "nano_intersection");
		owlGeomTypeSetClosestHit(geometryType, 0, module, "nano_closesthit");


		const auto missProgramVars =
			std::array{ OWLVarDecl{ "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgramData, color0) },
						OWLVarDecl{ "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgramData, color1) } };

		// ----------- create object  ----------------------------
		nanoContext.missProgram = owlMissProgCreate(context, module, "miss", sizeof(MissProgramData),
													missProgramVars.data(), missProgramVars.size());

		// ----------- set variables  ----------------------------
		owlMissProgSet3f(nanoContext.missProgram, "color0", owl3f{ .8f, 0.f, 0.f });
		owlMissProgSet3f(nanoContext.missProgram, "color1", owl3f{ .8f, .8f, .8f });


		// owlBuildProgramsDebug(context);
		owlBuildPrograms(context);
		owlBuildPipeline(context);
		// owlBuildPrograms(context)
		owlBuildSBT(context);
	}

	float computeStableEpsilon(float f)
	{
		return abs(f) * float(1. / (1 << 21));
	}

	float computeStableEpsilon(const vec3f v)
	{
		return max(max(computeStableEpsilon(v.x), computeStableEpsilon(v.y)), computeStableEpsilon(v.z));
	}


	std::filesystem::path b3dFilePath{};

} // namespace

auto NanoRenderer::onRender(const View& view) -> void
{

	// const auto& camera = view.cameras.front();
	// const auto aspect = view.colorRt.extent.width / static_cast<float>(view.colorRt.extent.height);

	// const auto projectionMatrix = glm::perspective(camera.FoV, aspect, 0.001f, 100.0f);
	// const auto viewMatrix = glm::lookAt(glm::vec3{ camera.origin.x, camera.origin.y, camera.origin.z },
	//									glm::vec3{ camera.at.x, camera.at.y, camera.at.z },
	//									glm::vec3{ camera.up.x, camera.up.y, camera.up.z });

	// const auto c00 = camera.at;
	// const auto origin = camera.origin;
	// float as = projectionMatrix[1][1]/projectionMatrix[0][0];
	// float tan = 1.0f/projectionMatrix[1][1];
	// vec3f cxx = tan * as * vec3f{  viewMatrix[0].x, viewMatrix[0].y, viewMatrix[0].z} ;
	// vec3f cyy = tan * vec3f{  viewMatrix[1].x, viewMatrix[1].y, viewMatrix[1].z};


	debugDraw().drawBox(nanoVdbVolume->worldAabb.center(), nanoVdbVolume->worldAabb.size(),
						owl::vec3f(0.1f, 0.82f, 0.15f));

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

	
	owlRayGenSetRaw(nanoContext.rayGen, "surfacePointers", cudaSurfaceObjects.data());

	owlMissProgSet3f(nanoContext.missProgram, "color0",
					 owl3f{ guiData.rtBackgroundColorPalette.color1[0], guiData.rtBackgroundColorPalette.color1[1],
							guiData.rtBackgroundColorPalette.color1[2] });
	owlMissProgSet3f(nanoContext.missProgram, "color1",
					 owl3f{ guiData.rtBackgroundColorPalette.color2[0], guiData.rtBackgroundColorPalette.color2[1],
							guiData.rtBackgroundColorPalette.color2[2] });


	const auto fbSize =
		owl2i{ static_cast<int32_t>(view.colorRt.extent.width), static_cast<int32_t>(view.colorRt.extent.height) };
	owlRayGenSet2i(nanoContext.rayGen, "frameBufferSize", fbSize);


	const auto& camera = view.cameras.front();
	const auto origin = vec3f{ camera.origin.x, camera.origin.y, camera.origin.z };
	auto camera_d00 = camera.at - origin;
	const auto wlen = length(camera_d00);
	owlRayGenSet3f(nanoContext.rayGen, "camera.position", reinterpret_cast<const owl3f&>(origin));
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


	owlRayGenSet3f(nanoContext.rayGen, "camera.dir00", reinterpret_cast<const owl3f&>(lower_left));
	owlRayGenSet3f(nanoContext.rayGen, "camera.dirDu", reinterpret_cast<const owl3f&>(horizontal));
	owlRayGenSet3f(nanoContext.rayGen, "camera.dirDv", reinterpret_cast<const owl3f&>(vertical));

	// owlRayGenSet3f(nanoContext.rayGen, "camera.dir00", reinterpret_cast<const owl3f&>(c00));
	// owlRayGenSet3f(nanoContext.rayGen, "camera.dirDu", reinterpret_cast<const owl3f&>(cxx));
	// owlRayGenSet3f(nanoContext.rayGen, "camera.dirDv", reinterpret_cast<const owl3f&>(cyy));
	// owlRayGenSet3f(nanoContext.rayGen, "camera.position", reinterpret_cast<const owl3f&>(origin));

	owlBuildSBT(nanoContext.context);
	owlAsyncLaunch2D(nanoContext.rayGen, view.colorRt.extent.width, view.colorRt.extent.height, nanoContext.lp);
	// owlLaunch2D(nanoContext.rayGen, view.colorRt.extent.width, view.colorRt.extent.height, nanoContext.lp);
	//  owlRayGenLaunch2D(nanoContext.rayGen, view.colorRt.extent.width, view.colorRt.extent.height);

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
	bar();
	log("[NanoRenderer] onInitialize!");
}

auto NanoRenderer::onDeinitialize() -> void
{
	log("[NanoRenderer] onDeinitialize!");
}

auto NanoRenderer::onGui() -> void
{
	ImGui::Begin("RT Settings");
	ImGui::SeparatorText("Data File (.b3d)");
	ImGui::InputText("##source", const_cast<char*>(b3dFilePath.string().c_str()), b3dFilePath.string().size(), ImGuiInputTextFlags_ReadOnly);
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

			const auto nanoFile = t.nanoVdbFile;
		}
		else
		{
			ImGui::TextColored(ImVec4{ 0.9f, 0.1f, 0.1f, 1.0f }, "Error: Can't load file!");
		}
	}
	ImGui::SeparatorText("Background Color Palette");
	ImGui::ColorEdit3("Color 1", guiData.rtBackgroundColorPalette.color1.data());
	ImGui::ColorEdit3("Color 2", guiData.rtBackgroundColorPalette.color2.data());

	static auto currentPath = std::filesystem::current_path();
	static auto selectedPath = std::filesystem::path{};

	const auto center = ImGui::GetMainViewport()->GetCenter();
	ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));

	// const auto root = std::filesystem::current_path().root_name();


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
		if (ImGui::BeginListBox("##dirs", ImVec2(ImGui::GetFontSize()*40, ImGui::GetFontSize()*16)))
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
					ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.1f,0.9f,0.1f,1.0f));
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
			if(!selectedPath.empty() != 0)
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
